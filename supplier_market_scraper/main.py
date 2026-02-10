import pandas as pd
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.linear_model import LinearRegression

# 1. DIRECTORY CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw_materials.csv"


def setup_environment():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"✅ Folder check: {DATA_DIR} is ready.")


# 2. DATA EXTRACTION (Simplified & Stable)
def harvest_market_data():
    print("🚀 Starting Chrome...")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in background
    driver = webdriver.Chrome(options=options)

    # We use the main catalogue page which is very stable
    target_url = "https://books.toscrape.com/catalogue/page-1.html"

    try:
        print(f"📡 Connecting to: {target_url}")
        driver.get(target_url)

        # Give the site a solid 5 seconds to load everything
        time.sleep(5)

        print("🔍 Searching for items...")
        scraped_items = []
        products = driver.find_elements(By.CLASS_NAME, 'product_pod')

        if not products:
            print("❌ Error: No items found on page. Check internet.")
            return pd.DataFrame()

        for item in products:
            # We use try/except inside the loop so one bad item doesn't kill the whole script
            try:
                # Finding the name and price using simple tags
                name = item.find_element(By.TAG_NAME, 'h3').text
                price_text = item.find_element(By.CLASS_NAME, 'price_color').text

                # Cleaning: Removing the '£' and converting to number
                price_val = float(price_text.replace('£', '').replace('$', ''))

                scraped_items.append({
                    'Material': name,
                    'Unit_Price': price_val,
                    'Weight_KG': round(price_val * 0.12, 2)
                })
            except:
                continue

        df = pd.DataFrame(scraped_items)
        print(f"✅ Success! Found {len(df)} items.")

        # Save the real data to your folder
        df.to_csv(RAW_DATA, index=False)
        return df

    finally:
        driver.quit()


# 3. MACHINE LEARNING (The Math)
def train_predictive_model(df):
    if df.empty:
        print("❌ No data to analyze.")
        return 0

    print("🧠 Training Linear Regression model...")

    # Target: Days_to_Delivery = (Price * 0.1) + (Weight * 0.5) + 2
    df['Lead_Time_Days'] = (df['Unit_Price'] * 0.1) + (df['Weight_KG'] * 0.5) + 2

    X = df[['Unit_Price', 'Weight_KG']]
    y = df['Lead_Time_Days']

    model = LinearRegression()
    model.fit(X, y)

    # Predicting lead time for a sample price of 50.0 and weight of 6.0
    prediction = model.predict([[50.0, 6.0]])
    return prediction[0]


# 4. RUN
if __name__ == "__main__":
    setup_environment()
    material_data = harvest_market_data()

    if not material_data.empty:
        result = train_predictive_model(material_data)
        print("\n" + "=" * 30)
        print(f"LIVE RESEARCH RESULTS")
        print(f"Items processed: {len(material_data)}")
        print(f"Predicted Lead Time: {result:.2f} Days")
        print("=" * 30)