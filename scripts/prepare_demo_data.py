from pathlib import Path
import pandas as pd

rows = [
    ("Battery life is excellent and lasts all day.", 0, "none"),
    ("The screen is bright and the product feels premium.", 0, "none"),
    ("The charging cable stopped working after two weeks.", 1, "quality_or_defect"),
    ("I received a damaged product and want a replacement.", 1, "damaged_item"),
    ("The item arrived late and the package was opened.", 1, "delivery_or_packaging"),
    ("The headphones sound great but the left earbud died.", 1, "quality_or_defect"),
    ("Fast delivery and exactly what I expected.", 0, "none"),
    ("The product is missing a component from the box.", 1, "missing_item"),
    ("The keyboard is comfortable and responsive.", 0, "none"),
    ("The item I received is the wrong model.", 1, "wrong_item"),
    ("Excellent build quality for the price.", 0, "none"),
    ("The zipper broke on the first day.", 1, "quality_or_defect"),
    ("Package arrived on time and undamaged.", 0, "none"),
    ("The charger was missing from the package.", 1, "missing_item"),
    ("The device works perfectly and setup was easy.", 0, "none"),
    ("I ordered black but received blue.", 1, "wrong_item"),
    ("The delivery was extremely late and support did not help.", 1, "delivery_or_packaging"),
    ("Very comfortable and good value.", 0, "none"),
    ("The box was crushed and the product was scratched.", 1, "damaged_item"),
    ("Battery performance is better than expected.", 0, "none"),
]

out = Path("data/processed/demo_reviews.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["review", "complaint", "complaint_category"]).to_csv(out, index=False)
print(f"Wrote {len(rows)} rows to {out}")
