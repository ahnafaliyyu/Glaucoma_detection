# %% [markdown]
# # RNFL v2 (KAGGLE): tugas biner / 4 kelas, input 128, fine-tuning ditingkatkan, ensemble, threshold LOFO, evaluasi tingkat mata
# - `TASK = 'binary'` : normal (kelas di `BINARY_NORMAL_CLASSES`) vs abnormal. `TASK = 'multiclass4'` : 4 kelas seperti sebelumnya.
# - Semua validasi per pasien (StratifiedGroupKFold). Oversampling hanya pada data train.
# - Prediksi out-of-fold disimpan -> dipakai untuk ensemble antar-arsitektur, threshold leave-one-fold-out, dan agregasi tingkat mata.
# - Dataset Kaggle: `label_sector_clean.csv` + `sector_clean.zip`.

# %%
import os, glob, re, time, shutil, zipfile, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, recall_score)
import matplotlib.pyplot as plt

START_TIME = time.time()
print(tf.__version__, tf.config.list_physical_devices('GPU'))   # harus ada GPU (Settings > Accelerator > GPU T4 x2)

INPUT_ROOT = '/kaggle/input'
CSV_NAME   = 'label_sector_clean.csv'
ZIP_NAME   = 'sector_clean.zip'
OUT_DIR    = '/kaggle/working/hasil_training/'
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# ---- Lokasi data (otomatis di /kaggle/input) ----
csv_found = glob.glob(f'{INPUT_ROOT}/**/{CSV_NAME}', recursive=True)
if not csv_found:
    print('Isi', INPUT_ROOT, ':'); print(*sorted(glob.glob(f'{INPUT_ROOT}/*') + glob.glob(f'{INPUT_ROOT}/*/*'))[:30], sep='\n')
    raise FileNotFoundError(f'{CSV_NAME} tidak ditemukan. Pastikan dataset sudah ditambahkan lewat Add Input.')
CSV_PATH = csv_found[0]

img_found = glob.glob(f'{INPUT_ROOT}/**/*_sector_0.png', recursive=True)
if img_found:                                    # Kaggle sudah mengekstrak zip menjadi folder gambar
    IMAGE_DIR = os.path.dirname(img_found[0]) + '/'
else:                                            # zip belum terekstrak -> ekstrak ke disk sementara
    zp = glob.glob(f'{INPUT_ROOT}/**/{ZIP_NAME}', recursive=True)
    if not zp: raise FileNotFoundError(f'Gambar maupun {ZIP_NAME} tidak ditemukan di {INPUT_ROOT}')
    with zipfile.ZipFile(zp[0]) as z: z.extractall('/tmp/rnfl_img/')
    IMAGE_DIR = os.path.dirname(glob.glob('/tmp/rnfl_img/**/*_sector_0.png', recursive=True)[0]) + '/'

# Lanjutkan dari output versi sebelumnya bila dilampirkan sebagai Input (Add Input > Notebook Output Files)
for p in glob.glob(f'{INPUT_ROOT}/**/hasil_cv_*.csv', recursive=True):
    src_dir = os.path.dirname(p)
    if 'hasil_training' in src_dir and not glob.glob(OUT_DIR + 'hasil_cv_*.csv'):
        for f in glob.glob(src_dir + '/hasil_cv_*.csv') + glob.glob(src_dir + '/pred_*.csv'):
            shutil.copy(f, OUT_DIR)
        print('Melanjutkan dari hasil sebelumnya:', src_dir)
        break

print('CSV   :', CSV_PATH)
print('Gambar:', IMAGE_DIR, '->', len(os.listdir(IMAGE_DIR)), 'file (harus 5180)')

# %%
df_raw = pd.read_csv(CSV_PATH)
print(df_raw.shape); display(df_raw.head()); print(df_raw['label'].value_counts().sort_index())

# %%
# ================= KONFIGURASI =================
FILE_COL, LABEL_COL = 'file_name', 'label'
GROUP_REGEX = r'^(fff_\d+_\d+)_'      # 1 pasien = mata kiri + kanan + semua scan ulang (sisi bisa l / r / u)
NAME_MAP = {1: 'kelas_1', 2: 'kelas_2', 3: 'kelas_3', 4: 'kelas_4'}   # arti kelas belum dipastikan

# ---- Tugas ----
TASK = 'binary'                 # 'binary' (normal vs abnormal) | 'multiclass4'
BINARY_NORMAL_CLASSES = [1]     # label yang dianggap "normal". ASUMSI: kelas 1 = normal -> konfirmasi ke pembimbing/dokumentasi dataset
EYE_ABNORMAL_MIN = 3            # mata dianggap abnormal bila >= K sektor abnormal. K=3 -> ~48% mata abnormal (seimbang). DEFINISI KLINIS: tentukan bersama dokter

# ---- Input & pelatihan ----
IMG_SIZE   = (128, 128)         # wedge asli hanya ~40-60 px; 224 = upsampling berlebih
BATCH_SIZE = 32
N_FOLDS    = 5
RUN_FOLDS  = None               # None = semua fold; [0] = uji cepat
SEED       = 42
EPOCHS_HEAD, EPOCHS_FT = 8, 25
PATIENCE   = 5
FT_LAYERS  = 80                 # jumlah layer terakhir backbone yang dibuka saat fine-tuning (BatchNorm tetap beku)
LR_HEAD, LR_FT = 1e-3, 3e-5     # fase 2: AdamW + cosine decay dari LR_FT
USE_SECTOR_INPUT = True
USE_MIXED_PRECISION = False     # True = lebih cepat di T4; jika error dtype kembalikan ke False
if USE_MIXED_PRECISION: keras.mixed_precision.set_global_policy('mixed_float16')

ARCHS     = ['ResNet50', 'MobileNetV2', 'InceptionV3']      # ensemble butuh >1 arsitektur dengan fold yang sama
SCENARIOS = [('os0.5', 0.5), ('os1.0', 1.0)]                # alpha oversampling (hanya data train)

RUN_TAG    = f'{TASK}_{IMG_SIZE[0]}'
RESULT_CSV = OUT_DIR + f'hasil_cv_{RUN_TAG}.csv'

# ---- MODE UJI CEPAT: True = 1 fold, 1 arsitektur, 1 skenario (beberapa menit) ----
QUICK_TEST = False
TIME_BUDGET_HOURS = 11.0        # setelah ini fold baru tidak dimulai (agar hasil tersimpan sebelum sesi 12 jam habis)
if QUICK_TEST:
    RUN_FOLDS, ARCHS, SCENARIOS = [0], ['MobileNetV2'], SCENARIOS[:1]
print('TASK =', TASK, '| RUN_TAG =', RUN_TAG, '| QUICK_TEST =', QUICK_TEST, '| ARCHS =', ARCHS,
      '| SCENARIOS =', [x[0] for x in SCENARIOS], '| RUN_FOLDS =', RUN_FOLDS)

# %%
df = df_raw.copy()
df['path'] = df[FILE_COL].astype(str).apply(lambda f: os.path.join(IMAGE_DIR, f))
missing = ~df['path'].apply(os.path.exists)
print('File hilang:', missing.sum(), 'dari', len(df))
df = df[~missing].reset_index(drop=True)

df['group'] = df[FILE_COL].astype(str).str.extract(GROUP_REGEX)[0]
assert df['group'].notna().all(), 'ada nama file yang tidak cocok GROUP_REGEX'
print('Jumlah pasien:', df['group'].nunique(), '| baris:', len(df))

# Sisi mata: l = kiri; r dan u = kanan
df['side'] = df[FILE_COL].astype(str).str.extract(r'^fff_\d+_\d+_([a-z])_')[0].map({'l': 'L', 'r': 'R', 'u': 'R'})
assert df['side'].notna().all(), 'ada sisi mata tak dikenal'
df['sec'] = df[FILE_COL].astype(str).str.extract(r'sector_(\d+)')[0].astype(int)
# Anatomis mata kiri = bayangan cermin mata kanan: sektor k (kiri) ~ sektor (10-k)%12 (kanan)
df['csec'] = np.where(df['side'] == 'L', (10 - df['sec']) % 12, df['sec'])
df['flip'] = (df['side'] == 'L').astype(int)
df['unit'] = df[FILE_COL].astype(str).str.extract(r'^(.*)_sector_')[0]      # 1 citra (1 scan 1 mata) = sampai 12 sektor
df['eye']  = df['group'] + '_' + df['side'].str.lower()                     # 1 mata fisik (gabungan scan ulang)
df['abn']  = (~df[LABEL_COL].isin(BINARY_NORMAL_CLASSES)).astype(int)

if TASK == 'binary':
    df['y'] = df['abn']; CLASS_NAMES = ['normal', 'abnormal']
else:
    le = LabelEncoder(); df['y'] = le.fit_transform(df[LABEL_COL])
    CLASS_NAMES = [NAME_MAP.get(c, str(c)) for c in le.classes_]
NUM_CLASSES = len(CLASS_NAMES)
print(CLASS_NAMES); print(df['y'].value_counts().sort_index())
print('Pasien per kelas:', {n: df.loc[df['y'] == i, 'group'].nunique() for i, n in enumerate(CLASS_NAMES)})
_u = df.groupby('unit')['abn'].sum()
print(f'Tingkat mata: {df["eye"].nunique()} mata, {len(_u)} citra; abnormal (>= {EYE_ABNORMAL_MIN} sektor abnormal): {(_u >= EYE_ABNORMAL_MIN).mean():.1%} citra')

# %%
PREPROCESS = {
    'MobileNetV2': keras.applications.mobilenet_v2.preprocess_input,
    'ResNet50':    keras.applications.resnet50.preprocess_input,
    'InceptionV3': keras.applications.inception_v3.preprocess_input,
}
BUILDERS = {
    'MobileNetV2': keras.applications.MobileNetV2,
    'ResNet50':    keras.applications.ResNet50,
    'InceptionV3': keras.applications.InceptionV3,
}

# Augmentasi ringan. TIDAK ada flip acak (flip hanya deterministik untuk menyamakan mata kiri dengan kanan).
augment = keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1, value_range=(0, 255)),
])

def crop_wedge(img):
    """Gambar asli = kanvas 448x448 hitam dengan wedge kecil. Crop ke bounding box wedge."""
    m = tf.reduce_max(img, axis=-1) > 12
    idx = tf.cast(tf.where(m), tf.int32)
    def _crop():
        y0 = tf.reduce_min(idx[:, 0]); x0 = tf.reduce_min(idx[:, 1])
        y1 = tf.reduce_max(idx[:, 0]) + 1; x1 = tf.reduce_max(idx[:, 1]) + 1
        return img[y0:y1, x0:x1, :]
    return tf.cond(tf.shape(idx)[0] > 20, _crop, lambda: img)

def make_ds(paths, labels, flips, secs, preprocess, training):
    def load(p, y, fl, sc):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = crop_wedge(tf.cast(img, tf.float32))
        img = tf.cond(fl > 0, lambda: tf.image.flip_left_right(img), lambda: img)
        img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
        return img, tf.one_hot(sc, 12), y
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, flips, secs))
    if training: ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    def prep(x, s_, y):
        if training: x = augment(x, training=True)
        x = preprocess(x)
        return ({'img': x, 'sec': s_}, y) if USE_SECTOR_INPUT else (x, y)
    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

def build_model(arch):
    base = BUILDERS[arch](include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    base.trainable = False
    inp = keras.Input(IMG_SIZE + (3,), name='img')
    x = layers.GlobalAveragePooling2D()(base(inp, training=False))
    if USE_SECTOR_INPUT:
        sec_in = keras.Input((12,), name='sec')
        x = layers.Concatenate()([x, sec_in]); inputs = {'img': inp, 'sec': sec_in}
    else:
        inputs = inp
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, out), base

def callbacks(ckpt, plateau=True):
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
          keras.callbacks.ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True)]
    if plateau:   # ReduceLROnPlateau tidak boleh dipakai bersama LR schedule (fase 2)
        cb.insert(1, keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7))
    return cb

def oversample(dtr, alpha=0.5, seed=SEED):
    """Hanya untuk data TRAIN. alpha=0 -> tanpa oversampling; alpha=1 -> semua kelas disamakan dengan kelas terbesar."""
    n = dtr['y'].value_counts(); nmax = n.max(); parts = []
    for c, nc in n.items():
        target = int(round(nc * (nmax / nc) ** alpha))
        g = dtr[dtr['y'] == c]
        parts.append(g if target <= nc else pd.concat([g, g.sample(target - nc, replace=True, random_state=seed)]))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

# %%
def run_cv(arch, scenario, alpha, done):
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(splitter.split(df['path'], df['y'], df['group'])):
        if RUN_FOLDS is not None and fold not in RUN_FOLDS: continue
        if (time.time() - START_TIME) / 3600 > TIME_BUDGET_HOURS:
            print(f'Batas waktu {TIME_BUDGET_HOURS} jam tercapai -> berhenti sebelum {scenario} | {arch} | fold {fold}. '
                  'Lanjutkan di versi baru (lampirkan output ini sebagai Input).'); return
        if (scenario, arch, fold) in done:
            print(f'lewati (sudah ada): {scenario} | {arch} | fold {fold}'); continue
        dtr, dva = df.iloc[tr], df.iloc[va]
        assert not (set(dtr['group']) & set(dva['group'])), 'kebocoran: pasien sama ada di train dan validasi'

        dfit = oversample(dtr, alpha)
        print(f'\n===== {RUN_TAG} | {scenario} | {arch} | fold {fold+1}/{N_FOLDS} | train={len(dfit)} (asli {len(dtr)}) val={len(dva)} =====')
        pre = PREPROCESS[arch]
        ds_tr = make_ds(dfit['path'].values, dfit['y'].values, dfit['flip'].values, dfit['csec'].values, pre, True)
        ds_va = make_ds(dva['path'].values, dva['y'].values, dva['flip'].values, dva['csec'].values, pre, False)

        keras.backend.clear_session()
        model, base = build_model(arch)
        ckpt = f'{OUT_DIR}{RUN_TAG}_{scenario}_{arch}_fold{fold}.weights.h5'

        # Fase 1: latih head saja
        model.compile(optimizer=keras.optimizers.Adam(LR_HEAD), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_HEAD, callbacks=callbacks(ckpt), verbose=2)

        # Fase 2: buka FT_LAYERS layer terakhir (BatchNorm tetap beku), AdamW + cosine decay
        base.trainable = True
        for l in base.layers[:-FT_LAYERS]: l.trainable = False
        for l in base.layers:
            if isinstance(l, layers.BatchNormalization): l.trainable = False
        steps = EPOCHS_FT * int(np.ceil(len(dfit) / BATCH_SIZE))
        sched = keras.optimizers.schedules.CosineDecay(LR_FT, decay_steps=steps)
        model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_FT, callbacks=callbacks(ckpt, plateau=False), verbose=2)

        prob = model.predict(ds_va, verbose=0); pred = prob.argmax(1); yt = dva['y'].values
        if NUM_CLASSES == 2:
            auc = roc_auc_score(yt, prob[:, 1]) if len(np.unique(yt)) == 2 else np.nan
        else:
            try: auc = roc_auc_score(yt, prob, multi_class='ovr', average='macro', labels=np.arange(NUM_CLASSES))
            except ValueError: auc = np.nan
        rec = recall_score(yt, pred, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)
        row = dict(scenario=scenario, arch=arch, fold=fold, acc=accuracy_score(yt, pred),
                   bal_acc=balanced_accuracy_score(yt, pred), macro_f1=f1_score(yt, pred, average='macro'), auc_ovr=auc)
        if NUM_CLASSES == 2: row['f1_abnormal'] = f1_score(yt, pred, zero_division=0)
        row.update({f'recall_{n}': r for n, r in zip(CLASS_NAMES, rec)})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ('scenario', 'arch')})

        # simpan SEGERA per fold: ringkasan + prediksi out-of-fold (untuk ensemble, threshold, tingkat mata)
        prev = pd.read_csv(RESULT_CSV) if os.path.exists(RESULT_CSV) else pd.DataFrame()
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(RESULT_CSV, index=False)
        pd.DataFrame(prob, columns=[f'p{i}' for i in range(NUM_CLASSES)]).assign(
            file_name=dva[FILE_COL].values, y=yt, fold=fold, arch=arch, scenario=scenario
        ).to_csv(f'{OUT_DIR}pred_{RUN_TAG}_{scenario}_{arch}_fold{fold}.csv', index=False)

# %%
# Jalankan. Urutan per arsitektur (semua skenario). Fold yang sudah ada dilewati otomatis (aman bila sesi terputus).
prev = pd.read_csv(RESULT_CSV) if os.path.exists(RESULT_CSV) else pd.DataFrame(columns=['scenario', 'arch', 'fold'])
done = set(zip(prev['scenario'], prev['arch'], prev['fold']))
for arch in ARCHS:
    for scenario, alpha in SCENARIOS:
        run_cv(arch, scenario, alpha, done)

# %%
# Ringkasan per (arsitektur, skenario): rata-rata antar-fold + simpangan baku
res = pd.read_csv(RESULT_CSV)
res = res[res['scenario'].isin([sc for sc, _ in SCENARIOS])]
metric_cols = [c for c in ['acc', 'bal_acc', 'macro_f1', 'auc_ovr', 'f1_abnormal'] if c in res.columns] + [c for c in res.columns if c.startswith('recall_')]
g = res.groupby(['arch', 'scenario'])
summary = g[metric_cols].mean().round(4)
for c in ['macro_f1', 'bal_acc', 'auc_ovr']: summary[c + '_std'] = g[c].std().round(4)
summary['n_fold'] = g.size()
display(summary); summary.to_csv(OUT_DIR + f'ringkasan_cv_{RUN_TAG}.csv')
print('Pembanding: tebakan kelas mayoritas -> akurasi', round(df['y'].value_counts(normalize=True).max(), 3), '| balanced acc', round(1 / NUM_CLASSES, 3))

# confusion matrix gabungan fold (dari prediksi out-of-fold)
combos = sorted({(r.scenario, r.arch) for r in res.itertuples()})
fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 3.8), squeeze=False)
for ax, (sc, ar) in zip(axes[0], combos):
    fs = glob.glob(f'{OUT_DIR}pred_{RUN_TAG}_{sc}_{ar}_fold*.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs]); pcols = [f'p{i}' for i in range(NUM_CLASSES)]
    cm = confusion_matrix(d['y'], d[pcols].values.argmax(1), labels=np.arange(NUM_CLASSES))
    ax.imshow(cm, cmap='Blues'); ax.set_title(f'{ar}\n{sc}', fontsize=9)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45); ax.set_yticklabels(CLASS_NAMES); ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES): ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR + f'confusion_matrix_{RUN_TAG}.png', dpi=150); plt.show()

# %%
# ===== Analisis out-of-fold: ensemble antar-arsitektur, threshold leave-one-fold-out (LOFO), evaluasi TINGKAT MATA =====
pcols = [f'p{i}' for i in range(NUM_CLASSES)]

def load_oof(scenario, arch):
    fs = sorted(glob.glob(f'{OUT_DIR}pred_{RUN_TAG}_{scenario}_{arch}_fold*.csv'))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None

def ensemble(frames):
    allp = pd.concat(frames); g = allp.groupby('file_name')
    out = g[pcols].mean(); out['y'] = g['y'].first(); out['fold'] = g['fold'].first(); out['n_models'] = g.size()
    out = out.reset_index(); return out[out['n_models'] == len(frames)]      # hanya baris yang dinilai SEMUA model

def lofo_pred(score, y, fold, grid):
    """Threshold dipilih (maks macro-F1) pada fold LAIN, lalu diterapkan ke fold ini -> tanpa kebocoran ke fold evaluasi."""
    pred = np.zeros(len(y), int)
    for f in np.unique(fold):
        te = fold == f; tr = ~te
        if tr.sum() == 0 or len(np.unique(y[tr])) < 2: th = grid[len(grid) // 2]
        else: th = grid[int(np.argmax([f1_score(y[tr], (score[tr] >= t).astype(int), average='macro') for t in grid]))]
        pred[te] = (score[te] >= th).astype(int)
    return pred

def bin_report(y, score, fold, grid, fixed_th):
    p0 = (score >= fixed_th).astype(int); pl = lofo_pred(score, y, fold, grid)
    return dict(n=len(y), pos_rate=round(float(np.mean(y)), 3),
                auc=round(roc_auc_score(y, score), 4) if len(np.unique(y)) == 2 else np.nan,
                acc_fixed=round(accuracy_score(y, p0), 4), f1m_fixed=round(f1_score(y, p0, average='macro'), 4),
                acc_lofo=round(accuracy_score(y, pl), 4), balacc_lofo=round(balanced_accuracy_score(y, pl), 4),
                f1m_lofo=round(f1_score(y, pl, average='macro'), 4), f1_pos_lofo=round(f1_score(y, pl, zero_division=0), 4))

def eye_table(d):
    """Agregasi sektor -> mata. Skor = ekspektasi jumlah sektor abnormal (dari p1). Label = >= EYE_ABNORMAL_MIN sektor abnormal (dari label OCT asli)."""
    m = d.merge(df[['file_name', 'unit', 'eye', 'abn']], on='file_name')
    u = m.groupby('unit').agg(eye=('eye', 'first'), fold=('fold', 'first'), n=('p1', 'size'), s=('p1', 'sum'), k=('abn', 'sum')).reset_index()
    u['s'] = u['s'] * 12 / u['n']
    e = u.groupby('eye').agg(fold=('fold', 'first'), score=('s', 'mean'), k=('k', 'max')).reset_index()
    e['y'] = (e['k'] >= EYE_ABNORMAL_MIN).astype(int); e['group'] = e['eye'].str.rsplit('_', n=1).str[0]
    return e

def boot_auc(e, B=500, seed=0):
    rng = np.random.default_rng(seed); groups = e['group'].unique(); by = {g: e[e['group'] == g] for g in groups}; vals = []
    for _ in range(B):
        s = pd.concat([by[g] for g in rng.choice(groups, len(groups))])
        if s['y'].nunique() == 2: vals.append(roc_auc_score(s['y'], s['score']))
    return np.round(np.percentile(vals, [2.5, 97.5]), 3) if vals else (np.nan, np.nan)

rows_s, rows_e = [], []
for sc, _ in SCENARIOS:
    named = {a: load_oof(sc, a) for a in ARCHS}; named = {a: f for a, f in named.items() if f is not None}
    if len(named) > 1: named['ENSEMBLE(' + '+'.join(named) + ')'] = ensemble(list(named.values()))
    for name, d in named.items():
        y, fold = d['y'].values, d['fold'].values
        if NUM_CLASSES == 2:
            rows_s.append(dict(skenario=sc, model=name, **bin_report(y, d['p1'].values, fold, np.linspace(0.05, 0.95, 91), 0.5)))
            e = eye_table(d)
            r = bin_report(e['y'].values, e['score'].values, e['fold'].values, np.linspace(0, 12, 49), EYE_ABNORMAL_MIN)
            lo, hi = boot_auc(e); r.update(auc_ci95=f'[{lo}, {hi}]')
            rows_e.append(dict(skenario=sc, model=name, **r))
        else:
            pr = d[pcols].values.argmax(1)
            rows_s.append(dict(skenario=sc, model=name, n=len(y), acc=round(accuracy_score(y, pr), 4),
                               bal_acc=round(balanced_accuracy_score(y, pr), 4), macro_f1=round(f1_score(y, pr, average='macro'), 4)))
print('=== TINGKAT SEKTOR (out-of-fold, validasi per pasien) ==='); sec_tab = pd.DataFrame(rows_s); display(sec_tab)
sec_tab.to_csv(OUT_DIR + f'analisis_sektor_{RUN_TAG}.csv', index=False)
if rows_e:
    print(f'=== TINGKAT MATA (abnormal = >= {EYE_ABNORMAL_MIN} sektor abnormal; threshold LOFO) ===')
    eye_tab = pd.DataFrame(rows_e); display(eye_tab); eye_tab.to_csv(OUT_DIR + f'analisis_mata_{RUN_TAG}.csv', index=False)
    print('Catatan: label tingkat mata diturunkan dari label sektor (OCT) dengan aturan K sektor; jumlah mata terbatas (~235) -> lihat auc_ci95.')

# %% [markdown]
# ## Cara membaca hasil
# - Bandingkan hanya angka dengan validasi per pasien. Jangan membandingkan tugas biner dengan tugas 4 kelas secara langsung (metriknya berbeda tingkat kesulitan).
# - `acc_fixed/f1m_fixed`: threshold tetap (0,5; atau K untuk tingkat mata). `*_lofo`: threshold dipilih di fold lain (tanpa bocor).
# - Tingkat mata: label berasal dari aturan K sektor abnormal (`EYE_ABNORMAL_MIN`) pada label OCT; K perlu definisi klinis. Laporkan `auc_ci95` (bootstrap per pasien).
# - Target >= 0,9 pada acc dan F1 tidak dijanjikan oleh skema ini. Jika tidak tercapai, laporkan sebagai baseline dengan batas data (127 pasien).
