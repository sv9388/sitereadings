Installation:
------------
1. Clone this repo using `git clone https://github.com/sv9388/sitereadings.git`
2. Install python3 and python3-pip and move to the root folder of this project
3. Run `pip3 install -r requirements.txt` to install other required dependencies
4. The main directory to upload your readings CSV is `<root_folder>/sitereadings` 
5. You can upload datewise folders and actual files within those folders.
6. NOTE: Order of columns should be preserved as is in the sample csv sheet.
7. You can upload them all in one shot. 
8. Once you have uploaded all the required files, please trigger the command ```python merger.py```

NOTE:
-----
* The db is a postgres db setup in localhost (ie. The local machine where the code is run). If you want to point it to a remote database, update the DB_URI parameter in the settings.py file to point to the remote hostname
