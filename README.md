# SITE READINGS #
## Folder Setup ##
This app has two folders. 
 * ems is the folder that contains a parallelized scraper to pull the site readings and dump it into the DB.
 * app_ems is the folder that contains the flask server which reads the contents of the DB and populates the charts and other details. 
 
## Installation ##
1. Clone this repository using the command `git clone https://github.com/sv9388/sitereadings.git` with your git credentials
2. Relocate to the root folder. By default, this will be sitereadings
3. Install `pip3` and `python3`
4. Run the command `pip3 install --user requirements.txt`

## Start the Server ##
Run the command `cd app_ems && sudo python3 manage.py runserver --host=0.0.0.0 --port=80`
The app is accessible via http://<server> (If you are setting it up locally server = localhost. If it is in AWS, server = Public IP of the machine)
