"""
Kitchen Compass AI - Automated Data Ingestion Script
Downloads and ingests sample NYC restaurant data
"""

import pandas as pd
import requests
from pathlib import Path
import time
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"[Step {step_num}] {text}")

def download_311_data(limit=5000):
    """Download NYC 311 complaints data"""
    print_step(1, "Downloading 311 Complaints Data...")
    
    try:
        # NYC Open Data API endpoint for 311 complaints
        url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
        params = {
            '$limit': limit,
            '$where': "complaint_type LIKE '%Food%' OR descriptor LIKE '%Restaurant%'"
        }
        
        print(f"   → Fetching {limit} food-related records...")
        df = pd.read_csv(url, params=params)
        
        # Clean and filter
        df = df.dropna(subset=['descriptor', 'created_date'])
        
        # Create output path
        output_path = Path('data/311_complaints.csv')
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"   ✅ Downloaded {len(df)} records")
        print(f"   → Saved to: {output_path}")
        return output_path, len(df)
        
    except Exception as e:
        print(f"   ❌ Error downloading 311 data: {e}")
        return None, 0

def download_inspection_data(limit=5000):
    """Download NYC restaurant inspection data"""
    print_step(2, "Downloading Restaurant Inspection Data...")
    
    try:
        # NYC Open Data API endpoint for restaurant inspections
        url = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv"
        params = {
            '$limit': limit,
            '$order': 'inspection_date DESC'
        }
        
        print(f"   → Fetching {limit} inspection records...")
        df = pd.read_csv(url, params=params)
        
        # Clean data
        df = df.dropna(subset=['dba', 'boro', 'inspection_date'])
        
        # Create output path
        output_path = Path('data/restaurant_inspections.csv')
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"   ✅ Downloaded {len(df)} records")
        print(f"   → Saved to: {output_path}")
        return output_path, len(df)
        
    except Exception as e:
        print(f"   ❌ Error downloading inspection data: {e}")
        return None, 0

def ingest_to_api(csv_path, api_url='http://127.0.0.1:8000'):
    """Ingest CSV file via API"""
    print(f"   → Uploading {csv_path.name} to vector store...")
    
    try:
        with open(csv_path, 'rb') as f:
            files = {'file': (csv_path.name, f, 'text/csv')}
            response = requests.post(f'{api_url}/api/ingest_csv', files=files, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Ingested {result.get('rows_ingested', 0)} rows")
                print(f"   → Total chunks in store: {result.get('total_chunks', 0)}")
                return True, result.get('total_chunks', 0)
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text[:200]}")
                return False, 0
                
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to API. Is the server running at {api_url}?")
        print(f"   → Start server with: python run.py")
        return False, 0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def verify_server(api_url='http://127.0.0.1:8000'):
    """Check if API server is running"""
    try:
        response = requests.get(f'{api_url}/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main execution flow"""
    print_header("Kitchen Compass AI - Data Ingestion")
    
    print("This script will:")
    print("  1. Download NYC 311 complaints data")
    print("  2. Download NYC restaurant inspection data")
    print("  3. Upload data to your vector store")
    print("\nNote: Make sure your server is running (python run.py)")
    
    # Ask for confirmation
    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("❌ Aborted by user")
        sys.exit(0)
    
    # Check if server is running
    print_step(0, "Checking API Server...")
    if not verify_server():
        print("   ❌ Server is not running!")
        print("   → Please start the server with: python run.py")
        print("   → Then run this script again")
        sys.exit(1)
    else:
        print("   ✅ Server is running")
    
    # Download data
    print_header("Downloading Data from NYC Open Data")
    
    complaints_path, complaints_count = download_311_data(limit=2000)
    time.sleep(1)  # Be nice to the API
    
    inspections_path, inspections_count = download_inspection_data(limit=2000)
    
    if not complaints_path and not inspections_path:
        print("\n❌ Failed to download any data")
        sys.exit(1)
    
    # Ingest data
    print_header("Ingesting Data into Vector Store")
    
    total_chunks = 0
    
    if complaints_path:
        print_step(3, "Ingesting 311 Complaints...")
        success, chunks = ingest_to_api(complaints_path)
        if success:
            total_chunks = chunks
        time.sleep(2)
    
    if inspections_path:
        print_step(4, "Ingesting Restaurant Inspections...")
        success, chunks = ingest_to_api(inspections_path)
        if success:
            total_chunks = chunks
        time.sleep(2)
    
    # Summary
    print_header("Ingestion Complete!")
    
    print("📊 Summary:")
    print(f"  • 311 Complaints: {complaints_count} records downloaded")
    print(f"  • Restaurant Inspections: {inspections_count} records downloaded")
    print(f"  • Total chunks in vector store: {total_chunks}")
    
    print("\n✅ Next Steps:")
    print("  1. Open http://127.0.0.1:8000")
    print("  2. Check 'Quick Stats' sidebar - Documents should show count")
    print("  3. Try asking: 'What are the most common violations in Manhattan?'")
    print("  4. Click '📊 Analytics' to see detailed stats")
    
    print("\n💡 Tips:")
    print("  • Download more data for better results (increase limit parameter)")
    print("  • Use the web interface to upload additional CSV files")
    print("  • Check DATA_INGESTION_GUIDE.md for more details")
    
    print("\n🎉 Your Kitchen Compass AI is ready to use!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)