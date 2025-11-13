# Тип задачі: класифікація зображень (визначення типу емоцій на обличчі)

# Використані 3 рівні моделей: 
- **Проста модель** — HOG + Logistic Regression (використала бібліотеку scikit-learn)
- **Середня модель** — Small CNN (PyTorch) 
- **Складна модель** — Vision Transformer (ViT) (бібліотека timm)

# Використаний датасет: **FER2013** посилання: https://www.kaggle.com/datasets/msambare/fer2013?resource=download вам треба його викачати вручну бо через обмеження GitHub датасет **не зберігається у репозиторії** . Після завантаження має бути файл: fer2013.zip.

# 1. Завантажте викачений датасет за посиланням вище, та закачайте його в папку проекту , щоб вона лежала в корені проекту разом з файлом train_fer.py, перейменуйте його на fer2013.zip

# 2. Створіть і увійдіть у віртуальне середовище:
python3 -m venv .venv
source .venv/bin/activate

# 3. Встановіть бібліотеки: 
pip install numpy pillow torch torchvision tqdm scikit-learn scikit-image timm

# 4. Прибираємо старі дані, якщо були
rm -rf data

# 5. Створюємо кореневу папку для даних
mkdir -p data

# 6. Розпаковуємо архів у data/
unzip -q fer2013.zip -d data/

 Тепер є два можливих варіанти структури після unzip:
 a) data/train, data/test
 b) data/fer2013/train, data/fer2013/test

# 7. Якщо train/test розпакувались прямо в data/, переносимо їх у data/fer2013
if [ -d "data/train" ] && [ -d "data/test" ]; then
    mkdir -p data/fer2013
    mv data/train data/fer2013/
    mv data/test  data/fer2013/
fi

# 8. Якщо всередині data/fer2013 ще одна папка fer2013 (типовий кейс деяких архівів),
  пересунемо все всередину "правильно"
if [ -d "data/fer2013/fer2013" ]; then
    mv data/fer2013/fer2013/train data/fer2013/
    mv data/fer2013/fer2013/test  data/fer2013/
    rm -rf data/fer2013/fer2013
fi

# 9. Створюємо validation з потрібними класами
mkdir -p data/fer2013/validation/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}

# 10. Переносимо по 100 зображень з train у validation для кожного класу
for d in Angry Disgust Fear Happy Neutral Sad Surprise; do
    ls "data/fer2013/train/$d" | head -n 100 | while read f; do
        mv "data/fer2013/train/$d/$f" "data/fer2013/validation/$d/"
    done
done


Після цього має з’явитися структура:
data/
└── fer2013/
    ├── train/
    ├── validation/
    └── test/

Якщо у вас немає unzip, встановіть:
MacOS: brew install unzip
Linux: sudo apt install unzip


# 11. Запуск моделей: 
обовʼязково знаходьтесь в корені проєкту: 

- Простий рівень: python train_fer.py --model simple --data_root data/fer2013
- Середній: python train_fer.py --model cnn --data_root data/fer2013
- Складний: python train_fer.py --model vit --data_root data/fer2013

Дочекайтесь закінчення кожної з них, проста пробіжить швидко , в середній 10 епох, в складній 1 епоха прописана

Можете для зручності запустити в трьох різних терміналах, і змінити кількість епох 

# 12. Якщо хочете протестувати на зображенні виконайте: 
python -c "from train_fer import predict_image_cnn; predict_image_cnn('data/fer2013/test/Happy/PrivateTest_647018.jpg')"    - але замість Happy/PrivateTest_647018.jpg , напишіть шлях до того яке хочете протестувати
