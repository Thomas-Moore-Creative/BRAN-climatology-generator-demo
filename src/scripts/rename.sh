#!/bin/bash

# Set the directory path where the files are located
directory="/path/to/files"

# Get the string to be inserted into the file names from command line argument
insert_string="$1"

# Loop through each file in the directory
for file in "$directory"/*; do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Get the file name and extension
        filename=$(basename "$file")
        extension="${filename##*.}"
        
        # Use sed to insert the new string after a number and underscore pattern
        new_filename=$(echo "$filename" | sed -r 's/([0-9]_)/\1'"$insert_string"'/')
        
        # Rename the file
        mv "$file" "$directory/$new_filename"
        
        echo "Renamed $filename to $new_filename"
    fi
done