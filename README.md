Installation:
------------
0. If it is a first time setup, see DB setup section.
1. Clone this repo using `git clone https://github.com/sv9388/sitereadings.git`
2. Install python3 and python3-pip and move to the root folder of this project
3. Run `pip3 install -r requirements.txt` to install other required dependencies
4. The main directory to upload your readings CSV is `<root_folder>/sitereadings` 
5. You can upload datewise folders and actual files within those folders.
6. NOTE: Order of columns should be preserved as is in the sample csv sheet.
7. You can upload them all in one shot. 
8. Once you have uploaded all the required files, please trigger the command ```python merger.py```

DB Setup:
---------
* Install postgres
* Run the commands in admin.sql in the project clone to setup the database. This will also create a new username password for this particular action.
* Login to postgres as this new user (siteruser by default) and run the commands in the file qschema.sql
* Note: The db is a postgres db setup in localhost (ie. The local machine where the code is run). If you want to point it to a remote database, update the DB_URI parameter in the settings.py file to point to the remote hostname
* These are a one-time action. Once this setup is done, the merger.py command can be executed smoothly.
