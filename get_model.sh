# Download dataset from googledrive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11jWU4oNkMVD7sikgEb9TpqDQK5uIcqQ-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11jWU4oNkMVD7sikgEb9TpqDQK5uIcqQ-" -O final.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./final.zip

# Remove the downloaded zip file
rm ./final.zip

### https://drive.google.com/open?id=11jWU4oNkMVD7sikgEb9TpqDQK5uIcqQ-
