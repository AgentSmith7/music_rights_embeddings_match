#!/usr/bin/env python3
"""
Spot check predictions by examining actual file contents.
"""

import tarfile
import zipfile
import io
import pandas as pd
from pathlib import Path

def parse_csv_sample(content: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=10, encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(io.BytesIO(content), nrows=10, encoding='latin-1', on_bad_lines='skip')
        except:
            return "PARSE ERROR"
    return f"Columns: {list(df.columns)}\nSample:\n{df.head(5).to_string()}"

def parse_excel_sample(content: bytes, ext: str) -> str:
    try:
        engine = 'xlrd' if ext == '.xls' else 'openpyxl'
        df = pd.read_excel(io.BytesIO(content), nrows=10, engine=engine)
    except:
        return "PARSE ERROR"
    return f"Columns: {list(df.columns)}\nSample:\n{df.head(5).to_string()}"

def parse_pdf_sample(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = doc[0].get_text()[:1000] if len(doc) > 0 else "EMPTY"
        doc.close()
        return text
    except:
        return "PARSE ERROR"

# Files to spot check from DURECO
dureco_checks = [
    ("DURECO/DURECO/SENA/SENA/Dureco/2020/2020-03/95504028.csv", "Sena Artist"),
    ("DURECO/DURECO/SENA/SENA/Dureco/2023/2023-03/95677064.pdf", "Sony Music Entertainment"),
    ("DURECO/DURECO/SENA/SENA/Dureco/2022/2022-03/95623850.csv", "PRS Writer"),
]

# Files to spot check from PubStrengholt  
pubstrengholt_checks = [
    ("Society/STIM/STIM/2022/2022-11/27649154.csv", "STIM"),
    ("Society/STIM/STIM/2022/2022-11/27649151.pdf", "Sony Music Entertainment"),
    ("Society/NCB/NCB/StrengholtNCBJune2022.xlsx", "ASCAP Publisher"),
    ("Society/SABAM/SABAM/2022/2022-09/Strengholt Sabam 2022-09.xlsx", "SABAM"),
    ("Society/MCPS_PRS/MCPS_PRS/2022/2022-09/PRS for Music - Strengholt Music Publishing - September 2022.xlsx", "PRS Writer"),
    ("Society/BUMA STEMRA/BUMA STEMRA/2022/2022-09/Strengholt Buma Stemra 2022-09.xlsx", "BUMA Writer"),
    ("Society/SACEM/SACEM/2022/2022-09/Strengholt Sacem 2022-09.xlsx", "SACEM Writer"),
]

print("=" * 80)
print("SPOT CHECK: Examining actual file contents vs predictions")
print("=" * 80)

# Check PubStrengholt files
print("\n\n### PUBSTRENGHOLT SOCIETY FILES ###\n")
try:
    with tarfile.open("/workspace/PubStrengholtSociety.tar.gz", "r:gz") as tf:
        for file_path, predicted_class in pubstrengholt_checks:
            print(f"\n{'='*60}")
            print(f"FILE: {file_path}")
            print(f"PREDICTED: {predicted_class}")
            print("-" * 60)
            
            try:
                member = tf.getmember(file_path)
                f = tf.extractfile(member)
                if f:
                    content = f.read()
                    ext = Path(file_path).suffix.lower()
                    
                    if ext == '.csv':
                        sample = parse_csv_sample(content)
                    elif ext in ['.xlsx', '.xls']:
                        sample = parse_excel_sample(content, ext)
                    elif ext == '.pdf':
                        sample = parse_pdf_sample(content)
                    else:
                        sample = content[:500].decode('utf-8', errors='ignore')
                    
                    print(f"CONTENT SAMPLE:\n{sample[:800]}")
            except KeyError:
                print(f"File not found in archive")
            except Exception as e:
                print(f"Error: {e}")
except Exception as e:
    print(f"Could not open PubStrengholt archive: {e}")

# Check DURECO files
print("\n\n### DURECO FILES ###\n")
try:
    with zipfile.ZipFile("/workspace/DURECO.zip", "r") as zf:
        for file_path, predicted_class in dureco_checks:
            print(f"\n{'='*60}")
            print(f"FILE: {file_path}")
            print(f"PREDICTED: {predicted_class}")
            print("-" * 60)
            
            # Need to find the file in nested zips
            try:
                # DURECO has nested structure
                for name in zf.namelist():
                    if name.endswith('.zip'):
                        nested_content = zf.read(name)
                        nested_zf = zipfile.ZipFile(io.BytesIO(nested_content))
                        for nested_name in nested_zf.namelist():
                            if file_path.endswith(nested_name) or nested_name.endswith(Path(file_path).name):
                                content = nested_zf.read(nested_name)
                                ext = Path(nested_name).suffix.lower()
                                
                                if ext == '.csv':
                                    sample = parse_csv_sample(content)
                                elif ext in ['.xlsx', '.xls']:
                                    sample = parse_excel_sample(content, ext)
                                elif ext == '.pdf':
                                    sample = parse_pdf_sample(content)
                                else:
                                    sample = content[:500].decode('utf-8', errors='ignore')
                                
                                print(f"CONTENT SAMPLE:\n{sample[:800]}")
                                break
                        nested_zf.close()
            except Exception as e:
                print(f"Error: {e}")
except Exception as e:
    print(f"Could not open DURECO archive: {e}")
