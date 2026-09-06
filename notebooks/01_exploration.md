# Exploration checklist

Before modeling, inspect:

1. Label distribution
2. Review length distribution
3. Duplicate reviews
4. Missing text
5. Category frequency
6. Lexical overlap between classes
7. Train/test leakage
8. Error slices by review length

Example questions:

- Are complaints much rarer than non-complaints?
- Does a simple keyword baseline already solve most examples?
- Which complaints are ambiguous?
- Are very short reviews harder to classify?
- Does adding aspect information improve complaint classification?

Do not tune on the test set.
