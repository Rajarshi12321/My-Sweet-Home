
# My Sweet Home

Welcome to the My Sweet Home repository, which is a House Price Prediction and Property Recommendation Flask app repository! This app is designed to predict house prices and recommend similar housing properties based on a saved dataset. Whether you are a homebuyer looking for your dream house or a real estate investor seeking lucrative opportunities, this app can help you make informed decisions.

## Table of Contents

- [My Sweet Home](#my-sweet-home)
  - [Table of Contents](#table-of-contents)
  - [Installation and Dependencies](#installation-and-dependencies)
  - [Working Directory](#working-directory)
  - [Working with the code](#working-with-the-code)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)



## Installation and Dependencies

These are some required packages for our program which are mentioned in the Requirements.txt file

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- catboost
- xgboost
- Flask
- dill
- requests
- beautifulsoup4
- bs4
- jinja2
- joblib
- librosa
- lxml




## Working Directory

```
📦Housing_predict_recommend
 ┣ 📂artifact
 ┃ ┗ 📜Dataset.csv
 ┣ 📂artifacts
 ┃ ┣ 📜data_preprocessed_recommend.csv
 ┃ ┣ 📜model.pkl
 ┃ ┣ 📜model_rent.pkl
 ┃ ┣ 📜preprocessor.pkl
 ┃ ┣ 📜processed_data.csv
 ┃ ┣ 📜recommend_data.csv
 ┃ ┗ 📜testing.py
 ┣ 📂catboost_info
 ┣ 📂housing
 ┣ 📂HousingProject.egg-info
 ┃ ┣ 📜dependency_links.txt
 ┃ ┣ 📜PKG-INFO
 ┃ ┣ 📜requires.txt
 ┃ ┣ 📜SOURCES.txt
 ┃ ┗ 📜top_level.txt
 ┣ 📂logs
 ┣ 📂NOTEBOOK
 ┃ ┣ 📂DATA
 ┃ ┃ ┗ 📜Scraped_Data.csv
 ┃ ┗ 📜indian-house-price-prediction.ipynb
 ┣ 📂src
 ┃ ┣ 📂components
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜data_transformation.cpython-39.pyc
 ┃ ┃ ┃ ┣ 📜model_trainer.cpython-39.pyc
 ┃ ┃ ┃ ┗ 📜__init__.cpython-39.pyc
 ┃ ┃ ┣ 📜data_ingestion.py
 ┃ ┃ ┣ 📜data_transformation.py
 ┃ ┃ ┣ 📜model_trainer.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂pipeline
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜predict_pipeline.cpython-39.pyc
 ┃ ┃ ┃ ┣ 📜scraping_pipeline.cpython-39.pyc
 ┃ ┃ ┃ ┗ 📜__init__.cpython-39.pyc
 ┃ ┃ ┣ 📜predict_pipeline.py
 ┃ ┃ ┣ 📜scraping_pipeline.py
 ┃ ┃ ┣ 📜train_pipeline.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂recommender
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜data_transformation_recommend.cpython-39.pyc
 ┃ ┃ ┃ ┣ 📜house_recommender.cpython-39.pyc
 ┃ ┃ ┃ ┗ 📜__init__.cpython-39.pyc
 ┃ ┃ ┣ 📜data_transformation_recommend.py
 ┃ ┃ ┣ 📜house_recommender.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜exception.cpython-39.pyc
 ┃ ┃ ┣ 📜logger.cpython-39.pyc
 ┃ ┃ ┣ 📜utils.cpython-39.pyc
 ┃ ┃ ┗ 📜__init__.cpython-39.pyc
 ┃ ┣ 📜exception.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📂static
 ┃ ┣ 📂css
 ┃ ┃ ┗ 📜signup.css
 ┃ ┗ 📂img
 ┃ ┃ ┣ 📜beautiful_house.jpg
 ┃ ┃ ┣ 📜default_pic.png
 ┃ ┃ ┗ 📜No Suitable house image found.png
 ┣ 📂templates
 ┃ ┣ 📜get_elements.py
 ┃ ┣ 📜home.html
 ┃ ┣ 📜index.html
 ┃ ┗ 📜testing.html
 ┣ 📜.gitignore
 ┣ 📜app.py
 ┣ 📜city_locality.npy
 ┣ 📜LICENSE
 ┣ 📜main.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜setup.py
 ```


## Working with the code


I have commented most of the neccesary information in the respective files.

To run this project locally, please follow these steps:-

1. Clone the repository:

   ```shell
   git clone https://github.com/Rajarshi12321/Housing_predict_recommend
   ```


2. Activating the env
  
    ```shell
    conda activate <your-env-name> 
    ```

3. Install the required dependencies by running:
   ```shell
    pip install -r requirements.txt.
    ``` 
   Ensure you have Python installed on your system (Python 3.9 or higher is recommended).<br />
   Once the dependencies are installed, you're ready to use the project.



4. Run the Flask app: Execute the following code in your terminal.
   ```shell  
    python app.py 
    ```
   

6. Access the app: Open your web browser and navigate to http://127.0.0.1:5000/ to use the House Price Prediction and Property Recommendation app.


## Usage
1. **House Price Prediction:** On the app's homepage, users can input the specific features of the house they are interested in. After submitting the details, the app will process the information and display the predicted price for the house.

2. **Property Recommendation:** Along with the house price predictions users will also get similar recommendation. The app will provide a list of 6 most similar properties that match the given criteria.

## Contributing
I welcome contributions to improve the functionality and performance of the app. If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bug fix.

2. Make your changes and ensure that the code is well-documented.

3. Test your changes thoroughly to maintain app reliability.

4. Create a pull request, detailing the purpose and changes made in your contribution.



## License
This project is licensed under the MIT License. Feel free to modify and distribute it as per the terms of the license.

I hope this README provides you with the necessary information to get started with the Housing Price Prediction and Recommending project. 


