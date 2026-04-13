# Exploratory Data Analysis Results

**Date:** April 12, 2026  
**Dataset:** Music Rights Training Data (trainData.tgz)

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Classes** | 378 |
| **Total Files** | 82,690 |
| **Total Size** | 368.9 GB |

---

## Class Distribution Statistics

| Metric | Value |
|--------|-------|
| Min files/class | 1 |
| Max files/class | 13,217 |
| Mean files/class | 218.8 |
| Median files/class | 18 |
| Classes < 5 files | 90 (23.8%) |
| Classes < 10 files | 148 (39.2%) |
| Classes > 100 files | 77 (20.4%) |
| Classes > 500 files | 29 (7.7%) |

**Observation:** Highly imbalanced dataset. Nearly 40% of classes have fewer than 10 files, while the top class (SoundExchange Artist) has 13,217 files.

---

## Top 30 Classes by File Count

| Rank | Class | Files |
|------|-------|-------|
| 1 | SoundExchange Artist | 13,217 |
| 2 | ASCAP Writer | 5,720 |
| 3 | BMI Writer | 5,686 |
| 4 | Sony Music Entertainment | 5,509 |
| 5 | Universal Music Group | 4,908 |
| 6 | Warner Music Group | 3,774 |
| 7 | Sony Music Publishing | 2,840 |
| 8 | SoundExchange Label | 2,700 |
| 9 | TuneCore | 2,077 |
| 10 | Kollective Neighbouring Rights | 1,758 |
| 11 | PRS Writer | 1,201 |
| 12 | Abramus Writer | 1,141 |
| 13 | Universal Music Publishing Group | 1,115 |
| 14 | Kobalt Music Publishing | 1,110 |
| 15 | Abramus Artist | 1,094 |
| 16 | BMG Publisher | 1,089 |
| 17 | BMI Publisher | 1,088 |
| 18 | UBC Writer | 1,087 |
| 19 | UBC Artist | 989 |
| 20 | The District | 948 |
| 21 | ASCAP Publisher | 881 |
| 22 | Warner Chappell | 829 |
| 23 | The Orchard | 822 |
| 24 | DistroKid | 759 |
| 25 | Stem Disintermedia | 728 |
| 26 | UBC Label | 664 |
| 27 | Royalty Exchange Artist | 564 |
| 28 | AWAL | 524 |
| 29 | Audiam | 507 |
| 30 | PRS Publisher | 497 |

---

## Bottom 10 Classes (Single File Each)

| Class | Files |
|-------|-------|
| Spotify Settlement | 1 |
| Never Say Die mechanicals | 1 |
| 800 Pound Gorilla Records | 1 |
| Position Music Writer | 1 |
| Songs of Innocence | 1 |
| KODA Publisher | 1 |
| 1091 Distribution | 1 |
| Kranky ltd. | 1 |
| Red Room Distribution | 1 |
| Klub Record | 1 |

---

## File Type Distribution

| Extension | Count | Size (GB) | % of Files |
|-----------|-------|-----------|------------|
| .csv | 52,376 | 313.02 | 63.3% |
| .xlsx | 27,001 | 50.16 | 32.7% |
| .pdf | 2,771 | 2.06 | 3.4% |
| .xls | 411 | 1.82 | 0.5% |
| .txt | 125 | 1.68 | 0.2% |
| .xlt | 4 | 0.00 | <0.1% |
| .tab | 2 | 0.16 | <0.1% |

**Observation:** CSV files dominate (63% of files, 85% of data size). Need robust parsers for CSV, XLSX, PDF, and XLS.

---

## Directory Structure

```
/workspace/training_data/extracted/
└── Fast/
    └── TrainData/
        └── RYLTY/
            └── Organizer/
                └── Statement/
                    ├── {CLASS_LABEL}/
                    │   └── {files...}
                    └── ...
```

**Class label extraction:** `path.parts[index_of('Statement') + 1]`

---

## Implications for Pipeline Design

1. **Class Imbalance:** Consider stratified sampling for train/val split
2. **Small Classes:** 90 classes have < 5 files - may need special handling
3. **File Types:** Need 4 parsers: CSV, XLSX, PDF, XLS (+ TXT, TAB as text)
4. **Size:** 369GB requires streaming approach, not in-memory loading
5. **Representation:** Full content summary (Option C) for semantic richness

---

## Next Steps

1. Build streaming file iterator
2. Implement extension-specific representation builders
3. Create 80/20 stratified train/val split
4. Run baseline with BGE-large embeddings
