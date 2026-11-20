from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json
from recipe_model import find_recipes, df

app = FastAPI(title="Fridge2Fork Recipe API", version="1.0")

class IngredientRequest(BaseModel):
    ingredients: List[str]
    diet_filter: Optional[str] = "No Restrictions"
    cuisine_filter: Optional[str] = "All Cuisines"
    exclude_allergens: Optional[List[str]] = []

@app.get("/")
def home():
    return {"status": "Fridge2Fork API is running!"}

@app.post("/recipes")
def get_recipes(req: IngredientRequest):
    """
    Takes detected/added ingredients from the iPhone app and returns matching recipes!
    """
    cleaned_ingredients = [item.strip().lower() for item in req.ingredients if item.strip()]

    make_now, missing_items = find_recipes(cleaned_ingredients, df)

    # 1. Allergen filtering
    allergen_map = {
        "Nuts": ["peanut", "almond", "walnut", "pecan", "cashew", "hazelnut", "pistachio", "nut", "nuts"],
        "Dairy": ["milk", "cheese", "butter", "cream", "yogurt", "sour cream", "ghee", "parmesan", "cheddar", "mozzarella"],
        "Gluten": ["flour", "wheat", "bread", "pasta", "breadcrumbs", "cereal", "spaghetti", "macaroni"],
        "Eggs": ["egg", "eggs", "mayonnaise", "mayo"],
        "Seafood": ["fish", "salmon", "tuna", "shrimp", "crab", "lobster", "cod", "clam", "seafood"]
    }

    excluded_words = set()
    for allergen in req.exclude_allergens:
        if allergen in allergen_map:
            excluded_words.update(allergen_map[allergen])
        else:
            excluded_words.add(allergen.lower().strip())

    if req.diet_filter in ("Vegetarian", "Vegan"):
        excluded_words.update(["chicken", "beef", "pork", "bacon", "turkey", "ham", "sausage", "meat", "pepperoni", "prosciutto", "lamb"])
    if req.diet_filter == "Vegan":
        excluded_words.update(["milk", "cheese", "butter", "cream", "egg", "eggs", "honey"])

    if excluded_words:
        def contains_allergens(recipe):
            all_text = " ".join(recipe.get('matched', []) + recipe.get('missing', [])).lower()
            return any(w in all_text for w in excluded_words)

        make_now = [r for r in make_now if not contains_allergens(r)]
        missing_items = [r for r in missing_items if not contains_allergens(r)]

    # 2. Cuisine filtering
    cuisine_keywords = {
        "Mexican": ["tortilla", "salsa", "cumin", "cilantro", "taco", "enchilada", "jalapeno", "guacamole", "quesadilla", "black beans", "mexican", "fajita"],
        "Italian": ["pasta", "parmesan", "basil", "oregano", "mozzarella", "marinara", "spaghetti", "lasagna", "ricotta", "pesto", "italian", "parmigiano", "fettuccine", "penne"],
        "Asian": ["soy sauce", "ginger", "sesame", "rice", "teriyaki", "noodle", "curry", "tofu", "miso", "sriracha", "bok choy", "asian", "ramen", "stir fry"],
        "Mediterranean": ["olive oil", "feta", "lemon", "oregano", "cucumber", "chickpea", "hummus", "pita", "tahini", "olive", "olives", "tzatziki", "mediterranean", "greek"],
        "American": ["burger", "bbq", "cheddar", "bacon", "potato", "pie", "macaroni", "biscuit", "coleslaw", "casserole", "meatloaf", "ranch"]
    }

    if req.cuisine_filter in cuisine_keywords:
        c_words = cuisine_keywords[req.cuisine_filter]
        def matches_cuisine(recipe):
            search_text = (recipe['title'] + " " + " ".join(recipe.get('matched', []) + recipe.get('missing', []))).lower()
            return any(kw in search_text for kw in c_words)

        c_now = [r for r in make_now if matches_cuisine(r)]
        c_missing = [r for r in missing_items if matches_cuisine(r)]
        if c_now or c_missing:
            make_now = c_now
            missing_items = c_missing

    # 3. Quick meals filter
    if req.diet_filter == "Quick Meals":
        make_now = [r for r in make_now if len(r['matched']) <= 4]
        missing_items = [r for r in missing_items if (len(r['matched']) + len(r['missing'])) <= 4]

    # Format recipes for JSON
    def format_for_json(recipe_list):
        formatted = []
        for r in recipe_list:
            ingr = r.get("ingredients", [])
            if isinstance(ingr, str):
                try:
                    ingr = json.loads(ingr)
                except:
                    ingr = [ingr]

            dirs = r.get("directions", [])
            if isinstance(dirs, str):
                try:
                    dirs = json.loads(dirs)
                except:
                    dirs = [dirs]

            formatted.append({
                "title": r["title"],
                "matched": r["matched"],
                "missing": r["missing"],
                "coverage": round(r["coverage"], 2),
                "ingredients": ingr,
                "directions": dirs
            })
        return formatted

    return {
        "count_make_now": len(make_now),
        "count_missing": len(missing_items),
        "make_now": format_for_json(make_now[:15]),
        "missing_items": format_for_json(missing_items[:20])
    }
