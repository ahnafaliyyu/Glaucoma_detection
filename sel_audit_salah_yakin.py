# ===== AUDIT label/registrasi: sektor yang salah diprediksi dengan YAKIN (bawa ke pembimbing/dokter) =====
from PIL import Image, ImageDraw
d = ensemble([load_oof(SCENARIOS[0][0], a) for a in ARCHS])
d['err_conf'] = np.where(d['y'] == 1, 1 - d['p1'], d['p1'])            # seberapa yakin model pada kelas yang SALAH
bad = d[d['err_conf'] > 0.75].merge(df[['file_name', 'group', 'side', 'sec', 'label']], on='file_name')
print(len(bad), 'sektor salah-yakin dari', len(d)); print(bad.groupby('label').size().rename('jumlah per label asli'))
bad = bad.sort_values('err_conf', ascending=False); bad.to_csv(OUT_DIR + 'audit_salah_yakin.csv', index=False)
W = 150; top = bad.head(24); canvas = Image.new('RGB', (W * 6, (W + 14) * 4), 'gray'); dr = ImageDraw.Draw(canvas)
for i, r in enumerate(top.itertuples()):
    im = Image.open(IMAGE_DIR + r.file_name).convert('RGB'); bb = im.getbbox()
    if bb: im = im.crop(bb)
    im.thumbnail((W, W)); x, y0 = (i % 6) * W, (i // 6) * (W + 14)
    canvas.paste(im, (x, y0 + 14)); dr.text((x + 2, y0 + 1), f'label={r.label} p_abn={r.p1:.2f}', fill='white')
canvas.save(OUT_DIR + 'audit_salah_yakin.png'); display(canvas)
