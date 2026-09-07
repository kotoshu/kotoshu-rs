#!/usr/bin/env python3
"""Build the LID parity fixture (plan 102).

Copies the registry lid.176 artifact pair (models repo,
kotoshu://models/lid/lid-176) into kotoshu/tests/fixtures/lid/ and
freezes the GEM detection outputs for the fixed multilingual corpus by
driving Kotoshu::Language::LanguageIdentifier over upstream lid.176.ftz
(the gem shells out to the fasttext bindings). The Rust integration
test (tests/lid_parity.rs) asserts the engine reproduces every frozen
code exactly and every score within the tolerance recorded in
parity.json.

The gem needs a python3 with `import fasttext` on PATH AND numpy < 2
(the bindings call np.array(..., copy=False), which numpy 2 rejects);
make one and put it first on PATH:

    python3 -m venv /tmp/lid-venv && /tmp/lid-venv/bin/pip install "numpy<2" fasttext-wheel
    PATH=/tmp/lid-venv/bin:$PATH scripts/make_lid_fixture.py

Without the gem the script still refreshes the artifact pair but keeps
the existing parity.json (it refuses to overwrite expectations it
cannot re-freeze).

Usage:
  scripts/make_lid_fixture.py \\
      --models-repo ../models-fasttext-onnx \\
      --gem-repo    ../kotoshu \\
      --out-dir     kotoshu/tests/fixtures/lid
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# The corpus is frozen here (and mirrored as the models repo probe,
# eval/corpora/lid_probe.jsonl): 13 languages x 4 natural prose
# sentences over four themes, plus empty / digits / mixed-punctuation
# edge cases.
CORPUS = [
    ("en-1", "en", "The committee will review the proposal before the end of the quarter."),
    ("en-2", "en", "She poured herself a cup of coffee and opened the morning newspaper."),
    ("en-3", "en", "Machine learning models need large amounts of labeled training data."),
    ("en-4", "en", "The old bridge across the river was rebuilt after the flood."),
    ("de-1", "de", "Der Ausschuss wird den Vorschlag vor Quartalsende prüfen."),
    ("de-2", "de", "Sie goss sich eine Tasse Kaffee ein und öffnete die Morgenzeitung."),
    ("de-3", "de", "Maschinelle Lernmodelle benötigen große Mengen annotierter Daten."),
    ("de-4", "de", "Die alte Brücke über den Fluss wurde nach dem Hochwasser neu gebaut."),
    ("es-1", "es", "El comité revisará la propuesta antes de fin de trimestre."),
    ("es-2", "es", "Se sirvió una taza de café y abrió el periódico de la mañana."),
    ("es-3", "es", "Los modelos de aprendizaje automático necesitan muchos datos etiquetados."),
    ("es-4", "es", "El viejo puente sobre el río fue reconstruido después de la inundación."),
    ("fr-1", "fr", "Le comité examinera la proposition avant la fin du trimestre."),
    ("fr-2", "fr", "Elle s'est versé une tasse de café et a ouvert le journal du matin."),
    ("fr-3", "fr", "Les modèles d'apprentissage automatique nécessitent beaucoup de données annotées."),
    ("fr-4", "fr", "Le vieux pont sur la rivière a été reconstruit après l'inondation."),
    ("it-1", "it", "Il comitato esaminerà la proposta prima della fine del trimestre."),
    ("it-2", "it", "Si è versata una tazza di caffè e ha aperto il giornale del mattino."),
    ("it-3", "it", "I modelli di apprendimento automatico richiedono molti dati etichettati."),
    ("it-4", "it", "Il vecchio ponte sul fiume è stato ricostruito dopo l'alluvione."),
    ("pt-1", "pt", "O comité analisará a proposta antes do fim do trimestre."),
    ("pt-2", "pt", "Serviu-se uma chávena de café e abriu o jornal da manhã."),
    ("pt-3", "pt", "Os modelos de aprendizagem automática precisam de muitos dados anotados."),
    ("pt-4", "pt", "A velha ponte sobre o rio foi reconstruída depois da cheia."),
    ("ru-1", "ru", "Комитет рассмотрит предложение до конца квартала."),
    ("ru-2", "ru", "Она налила себе чашку кофе и открыла утреннюю газету."),
    ("ru-3", "ru", "Модели машинного обучения требуют больших объёмов размеченных данных."),
    ("ru-4", "ru", "Старый мост через реку перестроили после наводнения."),
    ("ja-1", "ja", "委員会は四半期が終わる前に提案を審査します。"),
    ("ja-2", "ja", "彼女はコーヒーを一杯注いで、朝の新聞を開きました。"),
    ("ja-3", "ja", "機械学習モデルには大量の注釈付き訓練データが必要です。"),
    ("ja-4", "ja", "洪水のあと、川に架かる古い橋が建て替えられました。"),
    ("zh-1", "zh", "委员会将在本季度结束前审查这项提案。"),
    ("zh-2", "zh", "她给自己倒了一杯咖啡，翻开了晨报。"),
    ("zh-3", "zh", "机器学习模型需要大量带标注的训练数据。"),
    ("zh-4", "zh", "洪水过后，河上的老桥得到了重建。"),
    ("ko-1", "ko", "위원회는 분기가 끝나기 전에 제안을 검토할 것입니다."),
    ("ko-2", "ko", "그녀는 커피 한 잔을 따라서 아침 신문을 펼쳤습니다."),
    ("ko-3", "ko", "머신 러닝 모델은 대량의 레이블이 지정된 훈련 데이터가 필요합니다."),
    ("ko-4", "ko", "홍수가 지나간 후 강의 오래된 다리가 재건되었습니다."),
    ("ar-1", "ar", "ستراجع اللجنة المقترح قبل نهاية الربع."),
    ("ar-2", "ar", "صبت لنفسها فنجاناً من القهوة وفتحت صحيفة الصباح."),
    ("ar-3", "ar", "تحتاج نماذج التعلم الآلي إلى كميات كبيرة من البيانات الموسومة."),
    ("ar-4", "ar", "أعيد بناء الجسر القديم فوق النهر بعد الفيضان."),
    ("he-1", "he", "הוועדה תבחן את ההצעה לפני סוף הרבעון."),
    ("he-2", "he", "היא מזגה לעצמה כוס קפה ופתחה את עיתון הבוקר."),
    ("he-3", "he", "מודלים של למידת מכונה זקוקים לכמויות גדולות של נתונים מסומנים."),
    ("he-4", "he", "הגשר הישן על הנהר נבנה מחדש אחרי השיטפון."),
    ("hi-1", "hi", "समिति तिमाही के अंत से पहले प्रस्ताव की समीक्षा करेगी।"),
    ("hi-2", "hi", "उसने अपने लिए कॉफी का एक कप भरा और सुबह का अख़बार खोला।"),
    ("hi-3", "hi", "मशीन लर्निंग मॉडल को बड़ी मात्रा में अंकित प्रशिक्षण डेटा चाहिए।"),
    ("hi-4", "hi", "बाढ़ के बाद नदी के ऊपर पुराना पुल फिर से बनाया गया।"),
    ("edge-empty", None, ""),
    ("edge-digits", None, "12345 67890"),
    ("edge-mixed", None, "don't panic — it's just 42% of the übliche Ärger"),
]

def drive_gem(gem_repo: Path, ftz: Path) -> list[dict]:
    """Run the gem one-off: ruby -e over the corpus, model path as ARGV[0]."""
    corpus = json.dumps([{"id": i, "text": t} for i, _, t in CORPUS])
    script = r"""
require "json"
require "kotoshu"
corpus = JSON.parse(STDIN.read)
lid = Kotoshu::Language::LanguageIdentifier.new(model_path: ARGV[0], auto_download: false)
out = corpus.map do |sample|
  top = lid.detect(sample["text"], top_k: 1).first
  {"id" => sample["id"], "code" => top&.language, "score" => top&.confidence}
end
STDOUT.write(JSON.generate(out))
"""
    result = subprocess.run(
        ["ruby", "-e", script, str(ftz)],
        input=corpus,
        capture_output=True,
        text=True,
        cwd=gem_repo,
    )
    if result.returncode != 0:
        sys.exit(f"error: gem run failed:\n{result.stderr}")
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-repo", type=Path, default=Path("../models-fasttext-onnx"))
    parser.add_argument("--gem-repo", type=Path, default=Path("../kotoshu"))
    parser.add_argument("--ftz", type=Path, default=None, help="lid.176.ftz for the gem run")
    parser.add_argument("--out-dir", type=Path, default=Path("kotoshu/tests/fixtures/lid"))
    args = parser.parse_args()

    onnx = args.models_repo / "models" / "lid" / "lid.176.onnx"
    vocab = args.models_repo / "models" / "lid" / "lid.176.vocab.json"
    for path in (onnx, vocab):
        if not path.exists():
            sys.exit(f"error: {path} missing; run the models repo scripts/build_lid.py first")

    ftz = args.ftz or args.models_repo / "downloads" / "lid.176.ftz"
    if ftz.exists():
        expectations = drive_gem(args.gem_repo, ftz)
        codes = {e["id"]: e["code"] for e in expectations}
        wrong = [(i, x, codes.get(i)) for i, x, _ in CORPUS if x and codes.get(i) != x]
        if wrong:
            print(f"warning: {len(wrong)} labeled samples the gem detects differently: {wrong}", file=sys.stderr)
    else:
        print(f"[warn] {ftz} absent: keeping the existing parity.json expectations", file=sys.stderr)
        expectations = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lid.176.onnx").write_bytes(onnx.read_bytes())
    (args.out_dir / "lid.176.vocab.json").write_bytes(vocab.read_bytes())

    parity_path = args.out_dir / "parity.json"
    if expectations is None:
        if not parity_path.exists():
            sys.exit("error: no ftz and no existing parity.json - nothing to freeze")
        print("artifact pair refreshed; parity.json kept")
        return 0

    parity = {
        "comment": "Frozen gem detection outputs (Kotoshu::Language::LanguageIdentifier over upstream lid.176.ftz via the fasttext bindings); the Rust engine must reproduce every code and stay within tolerance on every score.",
        "tolerance": 1e-3,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": {
            "onnx_sha256": sha256(onnx.read_bytes()).hexdigest(),
            "vocab_sha256": sha256(vocab.read_bytes()).hexdigest(),
        },
        "samples": [
            {"id": i, "expected": x, "text": t, "code": e["code"], "score": e["score"]}
            for (i, x, t), e in zip(CORPUS, expectations)
        ],
    }
    parity_path.write_text(json.dumps(parity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"parity.json frozen over {len(CORPUS)} samples (tolerance 1e-3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
