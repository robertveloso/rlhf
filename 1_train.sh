export $(cat .env | xargs)

# chmod +x ./imdbbot/training/base/train_on_tensordock.sh
# ./imdbbot/training/base/train_on_tensordock.sh

python imdbbot/training/base/deploy_to_tensordock.py