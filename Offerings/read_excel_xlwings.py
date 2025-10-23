#!/usr/bin/env python3
"""
Script to read Excel file using xlwings (can read open files)
"""

import xlwings as xw
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    excel_file = "GAS LNG Use Cases v5.xlsm"
    
    try:
        print("="*60)
        print("ANALYZING: GAS LNG Use Cases v5.xlsm")
        print("="*60)
        
        # Check if file exists
        if not Path(excel_file).exists():
            print(f"Error: File '{excel_file}' not found!")
            return
        
        # Try to connect to the workbook (even if open)
        print("Connecting to Excel workbook...")
        
        try:
            # Try to connect to existing workbook first
            wb = xw.Book(excel_file)
            print("Connected to existing open workbook")
        except:
            # If not open, open it
            wb = xw.Book(excel_file)
            print("Opened workbook")
        
        # Get sheet names
        sheet_names = [sheet.name for sheet in wb.sheets]
        print(f"\nFound {len(sheet_names)} sheets:")
        for i, sheet in enumerate(sheet_names, 1):
            print(f"  {i}. {sheet}")
        
        # Analyze each sheet
        all_data = {}
        
        for sheet_name in sheet_names:
            print(f"\n{'='*50}")
            print(f"ANALYZING SHEET: {sheet_name}")
            print(f"{'='*50}")
            
            try:
                # Get the sheet
                sheet = wb.sheets[sheet_name]
                
                # Get used range
                used_range = sheet.used_range
                if used_range is None:
                    print("Sheet appears to be empty")
                    continue
                
                # Convert to pandas DataFrame
                data = used_range.value
                
                if data is None:
                    print("No data found in sheet")
                    continue
                
                # Handle single cell case
                if isinstance(data, (int, float, str)):
                    df = pd.DataFrame([[data]], columns=['Value'])
                else:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Set first row as column headers if it looks like headers
                    if len(df) > 1:
                        first_row = df.iloc[0]
                        if all(isinstance(x, str) for x in first_row if pd.notna(x)):
                            df.columns = first_row
                            df = df.drop(0).reset_index(drop=True)
                
                all_data[sheet_name] = df
                
                print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"Columns: {list(df.columns)}")
                
                # Show first few rows
                print(f"\nFirst 3 rows:")
                print(df.head(3).to_string())
                
                # Data types
                print(f"\nData types:")
                print(df.dtypes)
                
                # Missing values
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    print(f"\nMissing values:")
                    print(missing[missing > 0])
                else:
                    print(f"\nNo missing values found.")
                
                # Numeric columns analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    print(f"\nNumeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
                    print(f"Basic statistics:")
                    print(df[numeric_cols].describe())
                
                # Text columns analysis
                text_cols = df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    print(f"\nText columns ({len(text_cols)}): {list(text_cols)}")
                    for col in text_cols:
                        unique_count = df[col].nunique()
                        print(f"  {col}: {unique_count} unique values")
                        if unique_count <= 10:
                            print(f"    Values: {list(df[col].unique())}")
                
            except Exception as e:
                print(f"Error reading sheet '{sheet_name}': {e}")
        
        # Overall summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        
        total_rows = sum(df.shape[0] for df in all_data.values())
        total_cols = sum(df.shape[1] for df in all_data.values())
        
        print(f"Total sheets: {len(all_data)}")
        print(f"Total rows: {total_rows:,}")
        print(f"Total columns: {total_cols:,}")
        
        print(f"\nSheet details:")
        for sheet_name, df in all_data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            text_cols = df.select_dtypes(include=['object']).columns
            print(f"  {sheet_name}: {df.shape[0]} rows, {df.shape[1]} cols ({len(numeric_cols)} numeric, {len(text_cols)} text)")
        
        # Close workbook if we opened it
        try:
            wb.close()
        except:
            pass
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
