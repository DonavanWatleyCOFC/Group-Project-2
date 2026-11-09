import pandas as pd
import json

file_path = "/Users/adwoaaddai/Desktop/CSCI 270/Python/Group-Project-2/recipe_subset.csv"

df_preview = pd.read_csv(file_path, nrows=5)

print("Columns in dataset")
print(df_preview.columns.tolist()) #check the columns of the dataset


df = pd.read_csv(file_path, nrows=20000, usecols=["title", "ingredients", "NER", "directions"])
print(df["NER"].iloc[0])
print(type(df["NER"].iloc[0])) #check NER datatype. 

#Convert stringtype lists to python lists
def parse_list(item):
    try:
        return json.loads(item)
    except:
        return []
df["NER"] = df["NER"].apply(parse_list)
print(type(df["NER"].iloc[0]))
print(df["NER"].iloc[0])


def is_match(user_item: str, recipe_item: str) -> bool:
    user_item = user_item.lower().strip()
    recipe_item = recipe_item.lower().strip()
    
    #1. Exact match]
    if user_item == recipe_item:
        return True

    #2. Word matching
    user_words = set(user_item.split())
    recipe_words = set(recipe_item.split())

    #check if the user's item is one of the words in the recipe's item 
    if user_words.issubset(recipe_words) or recipe_words.issubset(user_words):
        return True
    return False

#find recipes
def find_recipes(available_ingredients, df_recipes):
    '''Take any list of ingredients and returns matching recipes from the dataset'''
    make_now = []
    missing_items= [] #things missing from user's ingredients

    for index, row in df_recipes.iterrows():
        title = row["title"]
        recipe_ingredients = row["NER"]

        #Skip if empty
        if not recipe_ingredients:
            continue
        
        matched = [] # for matching ingredients for the recipe
        missing = [] #for missing ingredients for the recipe

        for rec_item in recipe_ingredients:
            found = False
            for user_item in available_ingredients:
                if is_match(user_item, rec_item):
                    found = True
                    break
            if found:
                matched.append(rec_item)
            else:
                missing.append(rec_item)

        recipe_info = {
            "title" : title,
            "matched": matched,
            "missing": missing,
            "coverage": len(matched) / len(recipe_ingredients)
        }

        if len(missing) == 0:
            make_now.append(recipe_info)
        elif len(missing) in (1,2):
            missing_items.append(recipe_info)

    #Search through recipe in the datast
    return make_now, missing_items



#Test it with a sample list# Test it with a sample list!
sample_ingredients = ["eggs", "cheese", "butter", "milk", "salt", "bread"]
make_Now, missing = find_recipes(sample_ingredients, df)

print(f"\n--- Found {len(make_Now)} recipes you can MAKE NOW! ---")
for r in make_Now[:3]:  # print first 3
    print(f"{r['title']} -> Matched: {r['matched']}")

print(f"\n--- Found {len(missing)} recipes MISSING 1-2 ITEMS ---")
for r in missing[:3]:  # print first 3
    print(f"{r['title']} -> Missing: {r['missing']}")
