# Data

Large datasets are intentionally not committed to GitHub.

## Expected review schema

CSV columns:

```text
review,complaint,complaint_category
```

Where:
- `review`: raw customer review text
- `complaint`: 0/1
- `complaint_category`: category string

For aspect sentiment benchmarking, use a separate normalized file with:

```text
review,aspect,sentiment
```

Keep dataset licenses and attribution in this file before publishing.
