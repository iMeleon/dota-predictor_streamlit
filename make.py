import os

with open(r'C:\Users\scent\Desktop\streamlit\my_project\Procfile', "w") as file1:
    toFile = 'web: sh setup.sh && streamlit run first_app.py'
    file1.write(toFile)