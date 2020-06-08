# House sales prices in King County

A project on exploratory data analysis.

This was my first project at the neue fische Bootcamp Data Science. It was centered around exploratory data analysis techniques and simple predictive analysis using ordinary linear regression. After the bootcamp, the analysis was extended.

## Results

We have the following key insights:
- The distribution of house sale prices is left modal, with a median of about 0.5 million US Dollars.

![distribution of price][price]

- Location has a big impact on house sale price as can be visualized by the median house sale prices grouped by the zipcode:

![median prices grouped by zipcode][zipcodes]

The area with the highest housesale prices is Medina with zipcode 98039, a city in Eastside in the metropol region of Seattle.

![zipcodes map][zipcodes_map]

- There is a rough linear relationship between the living space area and the house sales price.

![living space][living_space]

- While there is a rough linear relationship between house condition and the house sales price, the quality of the interior (design/materials) has an exponential impact on the house sales price.

![condition grade][condition_grade]

- The better the view, the higher the house sales price. Most properties don't have an extraordinary view.

![view][view]

- If the house is on a waterfront, the median house sale price increases about 1 million US Dollars.

![waterfront][waterfront]

[price]: figures/price.svg "Distribution of price"
[zipcodes]: figures/zipcode.svg "House sale price medians grouped by zipcode"
[zipcodes_map]: figures/zipcode_map.svg "Map of zipcodes in King county, coloured by median house sale price"
[living_space]: figures/living_space.svg "Living space vs house sale prices"
[condition_grade]: figures/condition_grade.svg "House sale price medians grouped by condition resp grade"
[view]: figures/view.svg "House sale price medians grouped by view"
[waterfront]: figures/waterfront.svg "House sale price medians grouped by waterfront"


## Content

- [Part 1: Data mining](king_county_1_data_mining.ipynb)
- [Part 2: Data cleaning](king_county_2_data_cleaning.ipynb)
- [Part 3: Feature engineering](king_county_3_engineering.ipynb)
- [Part 4: Exploratory data analysis](king_county_4_exploratory_data_analysis.ipynb)
- [Part 5: Predictive analysis](king_county_5_predictive_analysis.ipynb)
- [Part 6: Visualization](king_county_6_visualization.ipynb)


## Future work

- try more regression algorithms
- try more ensemble methods
- try more feature selection methods
- try artificial neural networks