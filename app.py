from flask import Flask, render_template, request, redirect, url_for
import os
from pymongo import MongoClient
from werkzeug.utils import secure_filename

app = Flask(__name__)

# MongoDB Configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/'
app.config['DATABASE_NAME'] = 'pdf_app_db'

# File Upload Configuration
UPLOAD_FOLDER = 'database/upload/' #Folder for Uploaded PDF files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}  # Only allow PDF files

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# Initialize MongoDB
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['DATABASE_NAME']]
pdf_files_collection = db['pdf_files']  # Collection for individual files

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    file_paths = [] #List of files to store paths
    error_message = None #Check for any exceptions

    if request.method == 'POST':
        if 'pdf_files' in request.files:
            files = request.files.getlist('pdf_files')  # Get a list of uploaded files
            for file in files: #Iterate through the files
                if file and allowed_file(file.filename):
                    # Secure the filename, and set file path in database
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        # Save the file
                        file.save(file_path)

                        # Save file path to MongoDB
                        file_data = {
                            'filename': filename,
                            'file_path': file_path
                        }
                        result = pdf_files_collection.insert_one(file_data)
                        print(f"Saved file '{filename}' to MongoDB with ID: {result.inserted_id}")
                        file_paths.append(file_path)

                    except Exception as e:
                        error_message = f"An error occurred while saving {filename}: {e}"
                        print(error_message)

                else: #Error of allowed extension
                    error_message = f"File type not allowed, it must be .pdf"

        else:
            error_message = "No files were uploaded."

    return render_template('index.html', file_paths=file_paths, error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)