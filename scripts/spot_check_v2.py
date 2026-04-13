#!/usr/bin/env python3
"""
Spot check predictions by examining actual file contents.
Handles nested ZIP structures properly.
"""

import tarfile
import zipfile
import io
import pandas as pd
from pathlib import Path

def parse_csv_sample(content: bytes) -> str:
    try:
        # Try different delimiters
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(io.BytesIO(content), nrows=10, encoding='utf-8', on_bad_lines='skip', sep=sep)
                if len(df.columns) > 1:
                    return f"Columns: {list(df.columns)[:10]}\nSample:\n{df.head(3).to_string()}"
            except:
                pass
        df = pd.read_csv(io.BytesIO(content), nrows=10, encoding='latin-1', on_bad_lines='skip')
    except:
        return "PARSE ERROR"
    return f"Columns: {list(df.columns)[:10]}\nSample:\n{df.head(3).to_string()}"

def parse_excel_sample(content: bytes, ext: str) -> str:
    try:
        engine = 'xlrd' if ext == '.xls' else 'openpyxl'
        df = pd.read_excel(io.BytesIO(content), nrows=10, engine=engine)
    except:
        return "PARSE ERROR"
    return f"Columns: {list(df.columns)[:10]}\nSample:\n{df.head(3).to_string()}"

def parse_pdf_sample(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = doc[0].get_text()[:1500] if len(doc) > 0 else "EMPTY"
        doc.close()
        return text
    except:
        return "PARSE ERROR"

def extract_from_nested_zip(zf, target_filename):
    """Extract file from nested ZIP structure."""
    for name in zf.namelist():
        if name.endswith('.zip'):
            try:
                nested_content = zf.read(name)
                nested_zf = zipfile.ZipFile(io.BytesIO(nested_content))
                for nested_name in nested_zf.namelist():
                    if target_filename in nested_name:
                        content = nested_zf.read(nested_name)
                        nested_zf.close()
                        return content, nested_name
                # Check for deeper nesting
                result = extract_from_nested_zip(nested_zf, target_filename)
                nested_zf.close()
                if result:
                    return result
            except:
                pass
    return None, None

def extract_from_targz_nested(tf, target_filename):
    """Extract file from tar.gz with nested ZIPs."""
    for member in tf.getmembers():
        if member.name.endswith('.zip'):
            try:
                f = tf.extractfile(member)
                if f:
                    zip_content = f.read()
                    zf = zipfile.ZipFile(io.BytesIO(zip_content))
                    for name in zf.namelist():
                        if target_filename in name:
                            content = zf.read(name)
                            zf.close()
                            return content, name
                    zf.close()
            except:
                pass
    return None, None

print("=" * 80)
print("SPOT CHECK: Examining actual file contents vs predictions")
print("=" * 80)

# ============================================================
# DURECO CHECKS
# ============================================================
print("\n\n" + "=" * 80)
print("### DURECO HOLDOUT DATASET ###")
print("=" * 80)

dureco_checks = [
    ("95504028.csv", "Sena Artist", "SENA file from 2020"),
    ("95677064.pdf", "Sony Music Entertainment", "PDF from SENA 2023"),
    ("95623850.csv", "PRS Writer", "CSV from SENA 2022"),
    ("95752066.csv", "Sena Artist", "CSV from SENA 2024"),
]

try:
    with zipfile.ZipFile("/workspace/DURECO.zip", "r") as zf:
        for target_file, predicted, desc in dureco_checks:
            print(f"\n{'='*70}")
            print(f"FILE: {target_file} ({desc})")
            print(f"PREDICTED CLASS: {predicted}")
            print("-" * 70)
            
            content, found_path = extract_from_nested_zip(zf, target_file)
            if content:
                ext = Path(target_file).suffix.lower()
                if ext == '.csv':
                    sample = parse_csv_sample(content)
                elif ext in ['.xlsx', '.xls']:
                    sample = parse_excel_sample(content, ext)
                elif ext == '.pdf':
                    sample = parse_pdf_sample(content)
                else:
                    sample = content[:500].decode('utf-8', errors='ignore')
                
                print(f"Found at: {found_path}")
                print(f"\nCONTENT PREVIEW:\n{sample[:1000]}")
                
                # Verdict
                print(f"\n>>> VERDICT: ", end="")
                if "SENA" in found_path.upper() or "sena" in sample.lower():
                    if "Sena" in predicted:
                        print("CORRECT - File is from SENA, predicted Sena class")
                    else:
                        print(f"QUESTIONABLE - File is from SENA but predicted {predicted}")
                elif "stim" in sample.lower() or "STIM" in found_path:
                    if "STIM" in predicted:
                        print("CORRECT - File contains STIM data")
                    else:
                        print(f"CHECK - File may be STIM but predicted {predicted}")
                else:
                    print(f"MANUAL CHECK NEEDED")
            else:
                print("File not found in archive")
except Exception as e:
    print(f"Error opening DURECO: {e}")

# ============================================================
# PUBSTRENGHOLT CHECKS
# ============================================================
print("\n\n" + "=" * 80)
print("### PUBSTRENGHOLT SOCIETY HOLDOUT DATASET ###")
print("=" * 80)

pubstrengholt_checks = [
    ("27649154.csv", "STIM", "STIM CSV from 2022"),
    ("27649151.pdf", "Sony Music Entertainment", "STIM PDF from 2022"),
    ("StrengholtNCBJune2022.xlsx", "ASCAP Publisher", "NCB Excel file"),
    ("Strengholt Sabam 2022-09.xlsx", "SABAM", "SABAM Excel file"),
    ("PRS for Music - Strengholt Music Publishing - September 2022.xlsx", "PRS Writer", "PRS Excel file"),
    ("Strengholt Buma Stemra 2022-09.xlsx", "BUMA Writer", "BUMA STEMRA Excel file"),
    ("Strengholt Sacem 2022-09.xlsx", "SACEM Writer", "SACEM Excel file"),
    ("StrengholtPolarisDecember2023.xlsx", "SoundExchange Artist", "NCB Polaris file"),
]

try:
    with tarfile.open("/workspace/PubStrengholtSociety.tar.gz", "r:gz") as tf:
        for target_file, predicted, desc in pubstrengholt_checks:
            print(f"\n{'='*70}")
            print(f"FILE: {target_file} ({desc})")
            print(f"PREDICTED CLASS: {predicted}")
            print("-" * 70)
            
            content, found_path = extract_from_targz_nested(tf, target_file)
            if content:
                ext = Path(target_file).suffix.lower()
                if ext == '.csv':
                    sample = parse_csv_sample(content)
                elif ext in ['.xlsx', '.xls']:
                    sample = parse_excel_sample(content, ext)
                elif ext == '.pdf':
                    sample = parse_pdf_sample(content)
                else:
                    sample = content[:500].decode('utf-8', errors='ignore')
                
                print(f"Found at: {found_path}")
                print(f"\nCONTENT PREVIEW:\n{sample[:1000]}")
                
                # Verdict based on file path and content
                print(f"\n>>> VERDICT: ", end="")
                path_lower = found_path.lower() if found_path else ""
                sample_lower = sample.lower()
                
                if "stim" in path_lower:
                    if "STIM" in predicted:
                        print("CORRECT - File is from STIM folder, predicted STIM")
                    else:
                        print(f"CHECK - File is from STIM folder but predicted {predicted}")
                elif "ncb" in path_lower:
                    print(f"NCB file - predicted {predicted} (NCB is Nordic Copyright Bureau)")
                elif "sabam" in path_lower or "sabam" in sample_lower:
                    if "SABAM" in predicted:
                        print("CORRECT - File is SABAM data")
                    else:
                        print(f"CHECK - File appears to be SABAM but predicted {predicted}")
                elif "prs" in path_lower or "prs" in sample_lower:
                    if "PRS" in predicted:
                        print("CORRECT - File is PRS data")
                    else:
                        print(f"CHECK - File appears to be PRS but predicted {predicted}")
                elif "buma" in path_lower or "buma" in sample_lower or "stemra" in sample_lower:
                    if "BUMA" in predicted:
                        print("CORRECT - File is BUMA/STEMRA data")
                    else:
                        print(f"CHECK - File appears to be BUMA but predicted {predicted}")
                elif "sacem" in path_lower or "sacem" in sample_lower:
                    if "SACEM" in predicted:
                        print("CORRECT - File is SACEM data")
                    else:
                        print(f"CHECK - File appears to be SACEM but predicted {predicted}")
                else:
                    print(f"MANUAL CHECK NEEDED - predicted {predicted}")
            else:
                print("File not found in archive")
except Exception as e:
    print(f"Error opening PubStrengholt: {e}")

print("\n\n" + "=" * 80)
print("SPOT CHECK COMPLETE")
print("=" * 80)
