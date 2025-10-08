import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

titles, prices, ratings = [], [], []

for page in range(1,5) :
  url = f"https://www.amazon.com/s?k=laptops&page={page}"

  headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9",
      "Accept-Encoding": "gzip, deflate, br",
      "DNT": "1",
      "Connection": "keep-alive",
  }


  response = requests.get(url, headers=headers)
  soup = BeautifulSoup(response.content, "html.parser")





  for product in soup.find_all("div", {"data-component-type": "s-search-result"}) :
    title = product.h2.text.strip() if product.h2 else None
    titles.append(title)


    price_whole = product.find("span", class_="a-price-whole")
    price_fraction = product.find("span", class_="a-price-fraction")
    if price_whole and price_fraction:
        price = price_whole.text + price_fraction.text
    else:
        price = None
    prices.append(price)


    rating = product.find("span", class_="a-icon-alt")
    ratings.append(rating.text if rating else None)

  time.sleep(2)

valid_brands = ['hp', 'lenovo', 'acer', 'apple', 'samsung', 'nimo', 'asus', 'jumper', 'dell', 'microsoft']
brands = { 'hp' : 1 , 'lenovo' : 2 , 'acer' : 3 , 'apple' : 4 , 'samsung' : 5 , 'nimo' : 6 , 'asus' : 7 , 'jumper' : 8 , 'dell' : 9 }

df = pd.DataFrame({
    "Title": titles,
    "Price": prices,
    "Rating": ratings
})

df.dropna(inplace = True)
df['Rating'] = df['Rating'].apply(lambda x : float(x.strip().split()[0]))
df['Price'] = df['Price'].str.replace(',', '').astype(float)
df['Brand'] = df['Title'].apply(lambda x: x.strip().split()[0] if (isinstance(x, str) and x.strip().split() and x.strip().split()[0].lower() not in valid_brands) else (x.strip().split()[0] if isinstance(x, str) and x.strip().split() else None))
df['Description'] = df['Title'].apply(lambda x: None if not isinstance(x, str) or not x.strip().split() else (''.join(x.strip().split()[1:]) if x.lower().strip().split()[0] in valid_brands else x))


def extract_ram_storage(title):
    title = title.upper()

    # Find all memory-like terms (e.g. 8GB, 1TB)
    sizes = re.findall(r'(\d+(?:\.\d+)?\s*(?:GB|TB))', title)

    ram = None
    storage = None

    for s in sizes:
        if ("RAM" in title and s in title.split("RAM")[0]) or  ("UNIFIEDMEMORY" in title and s in title.split("UNIFIEDMEMORY")[0]) or  ("DDR5" in title and s in title.split("DDR5")[0])  or  ("LPDDR4" in title and s in title.split("LPDDR4")[0]) or  ("MEMORY" in title and s in title.split("MEMORY")[0])  :
            if 'GB' in s:
                ram = s.split('GB')[0]
            else:
                ram = float(s.split('TB')[0]) * 1024 # convert TB to GB
        elif "SSD" in title or "HDD" in title or "STORAGE" in title or "EMMC" in title or "CLOUDSTORAGE" in title:
            if 'GB' in s:
                storage = s.split('GB')[0]
            else:
                storage = float(s.split('TB')[0]) * 1024
    return pd.Series([ram, storage])

df[['RAM_GB', 'Storage_GB']] = df['Title'].apply(extract_ram_storage)
df.dropna(inplace = True)

df['RAM_GB'] = df['RAM_GB'].astype(int)
df['Storage_GB'] = df['Storage_GB'].astype(int)


df['BrandValid'] = df['Brand'].apply( lambda x : x if x and x.lower() in valid_brands else None)
df['Brand'] = df['BrandValid'].ffill()
df['Brand'] = df['Brand'].apply(lambda x : x.lower())
df['Brand'] = df['Brand'].replace( brands )
df.drop(columns = ['BrandValid', 'Title', 'Description'], inplace = True)
df.dropna(inplace = True)
df = df.reindex( columns = [ 'Brand', 'RAM_GB', 'Storage_GB', 'Rating', 'Price'] )

df.to_csv("amazon_laptops.csv", index=False)
print("Scraping done. Saved to amazon_laptops.csv")
print(df.to_string())

X = df[['Brand', 'RAM_GB', 'Storage_GB', 'Rating']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f'r2_score : {r2_score(y_test, y_pred)}')
print(f'mse : {mean_squared_error(y_test, y_pred)}')
#Predict the price for a laptop with Brand 4 (Apple), 4GB RAM, 512GB Storage, and a rating of 4.0
print(rf.predict([[4, 16, 512, 4.0]]))

plt.figure(figsize =(20, 10))
tree.plot_tree(rf.estimator_[0], feature_names = X_train.columns, filled = True, fontsize = 8)
plt.show()
