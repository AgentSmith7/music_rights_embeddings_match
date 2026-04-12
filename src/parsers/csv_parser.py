"""CSV document parser with schema and content extraction."""
import io
from typing import List, Optional
from loguru import logger

try:
    import pandas as pd
except ImportError:
    pd = None

from .base_parser import BaseParser, ParsedDocument, DocumentRepresentation, RepresentationType


class CSVParser(BaseParser):
    """Parser for CSV/Excel documents with semantic representation."""
    
    def __init__(
        self,
        max_sample_rows: int = 10,
        max_columns: int = 50,
        max_content_chars: int = 2000
    ):
        if pd is None:
            raise ImportError("pandas is required for CSV parsing. Install with: pip install pandas")
        
        self.max_sample_rows = max_sample_rows
        self.max_columns = max_columns
        self.max_content_chars = max_content_chars
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.csv', '.xlsx', '.xls', '.tsv']
    
    def parse(self, file_bytes: bytes, file_name: str) -> ParsedDocument:
        """Parse CSV/Excel into semantic representations."""
        representations = []
        
        try:
            file_ext = file_name.lower().split('.')[-1]
            
            if file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(file_bytes))
            elif file_ext == 'tsv':
                df = pd.read_csv(io.BytesIO(file_bytes), sep='\t')
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
            
            if df.empty:
                logger.warning(f"Empty dataframe from: {file_name}")
                return ParsedDocument(
                    file_name=file_name,
                    file_type=file_ext,
                    representations=[],
                    metadata={"error": "Empty dataframe"}
                )
            
            # Limit columns
            if len(df.columns) > self.max_columns:
                df = df.iloc[:, :self.max_columns]
            
            # 1. Schema representation
            schema_text = self._create_schema_representation(df)
            representations.append(DocumentRepresentation(
                text=schema_text,
                representation_type=RepresentationType.SCHEMA,
                metadata={
                    "num_columns": len(df.columns),
                    "num_rows": len(df),
                    "columns": list(df.columns)
                }
            ))
            
            # 2. Content representation (sample rows as natural language)
            content_text = self._create_content_representation(df)
            representations.append(DocumentRepresentation(
                text=content_text,
                representation_type=RepresentationType.CONTENT,
                metadata={"sample_rows": min(self.max_sample_rows, len(df))}
            ))
            
            # 3. Summary representation
            summary_text = self._create_summary_representation(df, file_name)
            representations.append(DocumentRepresentation(
                text=summary_text,
                representation_type=RepresentationType.SUMMARY,
                metadata={}
            ))
            
            # 4. Keywords (column names + unique categorical values)
            keywords = self._extract_csv_keywords(df)
            if keywords:
                representations.append(DocumentRepresentation(
                    text=keywords,
                    representation_type=RepresentationType.KEYWORDS,
                    metadata={}
                ))
            
            return ParsedDocument(
                file_name=file_name,
                file_type=file_ext,
                representations=representations,
                raw_text=df.to_string(),
                metadata={
                    "num_columns": len(df.columns),
                    "num_rows": len(df),
                    "columns": list(df.columns)
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing CSV {file_name}: {e}")
            return ParsedDocument(
                file_name=file_name,
                file_type="csv",
                representations=[],
                metadata={"error": str(e)}
            )
    
    def _create_schema_representation(self, df: 'pd.DataFrame') -> str:
        """Create a schema description of the dataframe."""
        lines = ["Table Schema:"]
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            total = len(df)
            
            if dtype == 'object':
                inferred_type = "text/string"
            elif 'int' in dtype:
                inferred_type = "integer"
            elif 'float' in dtype:
                inferred_type = "decimal/float"
            elif 'datetime' in dtype:
                inferred_type = "datetime"
            elif 'bool' in dtype:
                inferred_type = "boolean"
            else:
                inferred_type = dtype
            
            lines.append(f"- {col}: {inferred_type} ({non_null}/{total} non-null)")
        
        return "\n".join(lines)
    
    def _create_content_representation(self, df: 'pd.DataFrame') -> str:
        """Create natural language representation of sample rows."""
        lines = ["Sample data from table:"]
        
        sample_df = df.head(self.max_sample_rows)
        
        for idx, row in sample_df.iterrows():
            row_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    row_parts.append(f"{col}={val}")
            
            row_text = ", ".join(row_parts)
            if len(row_text) > 500:
                row_text = row_text[:500] + "..."
            lines.append(f"Row {idx}: {row_text}")
        
        result = "\n".join(lines)
        if len(result) > self.max_content_chars:
            result = result[:self.max_content_chars] + "..."
        
        return result
    
    def _create_summary_representation(self, df: 'pd.DataFrame', file_name: str) -> str:
        """Create a summary description of the table."""
        num_rows = len(df)
        num_cols = len(df.columns)
        columns = ", ".join(df.columns[:10])
        if len(df.columns) > 10:
            columns += f", ... and {len(df.columns) - 10} more columns"
        
        return f"Tabular data file '{file_name}' with {num_rows} rows and {num_cols} columns. Columns include: {columns}."
    
    def _extract_csv_keywords(self, df: 'pd.DataFrame') -> str:
        """Extract keywords from column names and categorical values."""
        keywords = list(df.columns)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 20:
                    keywords.extend([str(v) for v in unique_vals[:10]])
        
        return " ".join(set(str(k).lower() for k in keywords if len(str(k)) > 2))
