## Some information about this infrastructure
1. Please watch this [short video](https://www.youtube.com/watch?v=04bJjSSjvg8) first to know how this infrastructure works. 
2. Researchers are able to request from each data party:
  ```shell
    - an overview of the dataset
    - check missing values
    - correlation matrixs
    - feature distributions
    - box-plots 
    - relation plots (two/three features)
  ```
2. Based on the data information, researchers can revise requested features and provide parameters of analysis models. We support classification and regression models (based on scikit learn library) including: 

  ```shell
    - Linear regression 
    - Lasso regresssion 
    - Logistic regression
    - Support vector machine
    - Gradient boosting
    - Decision Tree
    - Random Forest
    - Bernoulli Naive Bayes
    - Gaussian Naive Bayes
  ```

3. We support **Precision, Recall, F1-Score, ROC Area**, as evaluation methods for classification, **mean absolute error, mean squared error, mean squared log error, and R2** as regression evaluation methods.

## Try out locally ##

Please note that this repository is an extension of [Datasharing](https://gitlab.com/OleMussmann/DataSharing)

### 1. Setup stations at data parties ###
1. Go to **containers/createContainer/baseContainer**  and build the base image (contains Python 3.6 and libraries) by running:
  ```shell
  # docker rmi datasharing/base # (uncomment this line if you built this image before)
  docker build -t datasharing/base .
  ```

2. Go to **containers/createContainer/Party_1_Container/StepRequest**  (and "Party_2_Container" later on) and build **Docker Image 1** for extracting basic information about data. Run 
  ```shell
  # docker rmi datasharing/"your_party_name"_basicinfo # (uncomment this line if you built this image before)

  docker build -t datasharing/*your_party_name*_basicinfo .
  ```

3. To configure what information you want to extract, please edit **request.json** file. 

  ```shell
  {
      "data_file": "Diabetes_PartA.csv",
      "selected_features": "ALL",
      "check_missing": true,
      "data_description": true,
      "correlation_matrix": true,
      "distribution_plot": true,
      "distribution_feature": "ALL",

      "Box_plot": true,
      "Box_plot_feature":[["Employment_status","weight","N_Diabetes_WHO_2"]],

      "Cat_Num_plot": true,
      "Cat_Num_feature":[["Employment_status","bmi","N_Diabetes_WHO_2"]],

      "Num_Num_Plot": true,
      "Num_Num_feature":[["height","weight","N_Diabetes_WHO_2"]]
  }
  ```

4. Execute the Docker Container:

   - Linux/macOS:

   ```shell
   docker run --rm --add-host dockerhost:192.168.65.2 \
   -v "$(pwd)/request.json:/request.json" \
   -v "$(pwd)/Diabetes_PartA.csv:/Diabetes_PartA.csv" \
   -v "$(pwd)/output:/output" datasharing/*your_party_name*_basicinfo
   ```

   - Windows:

   ```shell
   docker run --rm --add-host dockerhost:10.0.75.1 \
   -v "%cd%/request.json:/request.json" \
   -v "%cd%/Diabetes_PartA.csv:/Diabetes_PartA.csv" \
   -v "%cd%/output:/output" datasharing/*your_party_name*_basicinfo
   ```

5. You will see all results in the output folder.

6. Now go to **.../Party_1_Container/StepEnc** folder, and edit **input.json** file

   ```shell
   {   "party_name": *your_party_name*,
       "data_file": "Diabetes_PartA.csv", 
       "salt_text": *salt*, # data parties agree on one salt
       "id_feature": ["housenum", "zipcode", "date_of_birth", "sex"]} # identifiable features for linking purpose
   ```

7. build **Docker Image 2** - Pseudonymization and encryption
  ```shell
  # docker rmi datasharing/"your_party_name"_enc #(uncomment this line if you built this image before)

  docker build -t datasharing/*your_party_name*_enc .
  ```


### 3. Setup another station as Trust Secure Environment (TSE) ###

1. Go to containers/TSEImage and run:
```shell 
# docker rmi datasharing/tse (uncomment this line if you built this image before)

docker build -t datasharing/tse .
```

**Now, all parties are ready**

### 4. Start the communication channel (on localhost(0.0.0.0:5001)) ###
1. Go to Local_PyTaskManager folder and run in terminal: 
  
```shell
docker build -t fileservice .
```

2. After building the image, run
- Linux/macOS:
```shell
docker run --rm -p 5001:5001 -v "$(pwd)/storage:/storage" fileservice
```

- Windows:
```shell
docker run --rm -p 5001:5001 -v "%cd%/storage:/storage" fileservice
```

**3.	If connection is failed, alternative way:**

- Install python 3.6 + and pip on your machine

- Install Flask library and run fileservice locally (not in a container)

- ```shell
  pip install flask 
  python FileService.py
  ```

### 5. Data parties encrypt data files and send to TSE ###

1. Go to each party's folder **.../Party_1_Container/StepEnc** (e.g., Party_1_Container), execute encryption Docker Containers:

   - Linux/macOS

   ```shell
   docker run --rm --add-host dockerhost:192.168.65.2 \
   -v "$(pwd)/Diabetes_PartA.csv:/Diabetes_PartA.csv" \
   -v "$(pwd)/input.json:/input.json" \
   -v "$(pwd)/encryption:/encryption" datasharing/*your_party_name*_enc
   ```

   - Windows

   ```shell
   docker run --rm --add-host dockerhost:10.0.75.1 \
   -v "%cd%/Diabetes_PartA.csv:/Diabetes_PartA.csv" \
   -v "%cd%/input.json:/input.json" \
   -v "%cd%/encryption:/encryption" datasharing/*your_party_name*_enc
   ```

     A **your_party_name_key.json** file will be generated in a new **encryption** folder. It contains: UUID of data file, verify key, and encryption key. These keys need to be send to TSE
    ```json
    {
      "party_1fileUUID": "xxxxx", 
      "party_1encryptKey": "xxxxx",
      "party_1verifyKey": "xxxxx"
    }
    ```



### 6. Execution at TSE ###
1. Go to ***containers/TSEImage*** and edit **security_input.json**:
  
  ```json
  {
    "parties": ["party_1","party_2"],
    "party_1fileUUID": "xxxxx", 
    "party_1encryptKey": "xxxxx",
    "party_1verifyKey": "xxxxx",
    "party_2fileUUID": "yyyyy",
    "party_2encryptKey": "yyyyy",
    "party_2verifyKey": "yyyyy",
  }
  ```

2. Configure analysis model parameters in **analysis_input.json** file:

   ```shell
   
   {   
       "taskName": ["analysis_test"],
       "check_missing": [true],
       "correlation_matrix": [true],
       
       "task": ["regression"],
       "model": ["linear regression"], 
       "parameters": [{"fit_intercept":true, "normalize":false, "copy_X":true}],
   
       "training_features":[["num_lab_procedures", "num_medications", "time_in_hospital"]], 
       "target_feature":["KOSTEN_FARMACIE"], 
   
       "evaluation_methods": [["neg_mean_absolute_error","neg_mean_squared_error","neg_mean_squared_log_error","r2"]], 
       "k_fold": [10]
   }
   ```

   

1. After configuring, execute the Docker Container at TSE:
- Linux/macOS:
```shell
docker run --rm --add-host dockerhost:192.168.65.2 \
-v "$(pwd)/output:/output" \
-v "$(pwd)/analysis_input.json:/analysis_input.json" \
-v "$(pwd)/security_input.json:/security_input.json" datasharing/tse
```

- Windows:
```shell
docker run --rm --add-host dockerhost:10.0.75.1 \
-v "%cd%/output:/output" \
-v "%cd%/analysis_input.json:/analysis_input.json" \
-v "%cd%/security_input.json:/security_input.json" datasharing/tse
```

4. All results are generated at TSE Output folder.  

## Data information ##

We used two public data sets to simulate real-life case for party A and party B

Data source for party A: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008 

Data source for party B: https://www.vektis.nl/intelligence/open-data 

## Contact ##

Chang Sun <chang.sun@maastrichtuniversity.nl>
Johan van Soest <johan.vansoest@maastro.nl>

