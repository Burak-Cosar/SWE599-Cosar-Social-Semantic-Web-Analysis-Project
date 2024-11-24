# SWE599-Cosar-Social-Semantic-Web-Analysis-Project

## Description
The purpose of this tool is to enable users to make social media analysis based on their platform, context and time period preferences, providing them with multiple output and visualization options.

For the scope of this project, considering the limitations of various social media platforms, the system connection is only established with Reddit. 

## Setup

Project setup:
`python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt`

.env file (Required):
`# .env file. Developers need to fill in their Reddit Developer credentials for usage.
CLIENT_ID = your_client_id
SECRET_ID = your_secret_id
USERNAME = your_username
PASSWORD = your_password`

Run server:
`python3 manage.py makemigrations &&
python3 manage.py migrate &&
python3 manage.py runserver`
