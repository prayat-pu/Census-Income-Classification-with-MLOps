# Census-Income-Classification-with-MLOps
This initiative is a Udacity coursework assignment aimed at developing a Census Income Classification API system. We are implementing continuous integration and continuous deployment (CI/CD) methodologies for this project. Additionally, the system is hosted on Render, and utilizes FastAPI for API development.


## Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.


## Data
* Download census.csv from the data folder in the starter repository.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

## Model
The model utilized in this research is the random forest. Further information about the model can be found in the model card [see model_card.md].


## API Creation
To create an API, we utilized the FastAPI framework. In addition, we also implement the test of our API with Pytest.


## API Deployment
For this project, we have set up automatic deployment through render.com. This ensures that the project is automatically deployed with the latest commit from our GitHub repository.

