# Тип задачі: класифікація зображень (визначення типу емоцій на обличчі)

# Використані 3 рівні моделей: 
- **Проста модель** — HOG + Logistic Regression (використала бібліотеку scikit-learn)
- **Середня модель** — Small CNN (PyTorch) 
- **Складна модель** — Vision Transformer (ViT) (бібліотека timm)

### Використаний датасет: **FER2013** посилання: https://www.kaggle.com/datasets/msambare/fer2013?resource=download вам треба його викачати вручну бо через обмеження GitHub датасет **не зберігається у репозиторії** . 

# 1. Завантажте викачений датасет за посиланням вище, та закачайте його в папку проекту , щоб вона лежала в корені проекту разом з файлом train_fer.py, ПЕРЕЙМЕНУЙТЕ його на fer2013.zip

#### оскільки в мене macOS, команди на нього є точними та працюючими, на Windows я шукала відповідники, надіюсь вони теж працюють, на жаль не маю змоги це перевірити, якщо у вас саме ця система

# 2. Створіть і увійдіть у віртуальне середовище:

### Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
```

# 3. Встановіть бібліотеки: 
### Mac / Linux
```bash
pip install numpy pillow torch torchvision tqdm scikit-learn scikit-image timm
```

### Windows
```powershell
pip install numpy pillow torch torchvision tqdm scikit-learn scikit-image timm
```

# 4. Прибираємо старі дані, якщо були

### Mac / Linux
```bash
rm -rf data
```
### Windows
```powershell
Remove-Item -Recurse -Force data
```

# 5. Створюємо кореневу папку для даних

### Mac / Linux
```bash
mkdir -p data
```

### Windows
```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
```

# 6. Розпаковуємо архів у data/

### macOS / Linux:
```bash
unzip -q fer2013.zip -d data/
```

### Windows: 
```powershell
Expand-Archive -Path fer2013.zip -DestinationPath data
```
### або краще: 
```powershell
Expand-Archive -Path fer2013.zip -DestinationPath data -Force
```
Якщо у вас немає unzip, встановіть:
### Mac
```bash
brew install unzip
```

### Linux
```bash
sudo apt install unzip
```


 Тепер є два можливих варіанти структури після unzip:
 a) data/train, data/test
 b) data/fer2013/train, data/fer2013/test


# 7. Якщо train/test розпакувались прямо в data/, переносимо їх у data/fer2013

### macOS / Linux:
```bash
if [ -d "data/train" ] && [ -d "data/test" ]; then
    mkdir -p data/fer2013
    mv data/train data/fer2013/
    mv data/test  data/fer2013/
fi
```

### Windows:
```powershell
if ((Test-Path "data/train") -and (Test-Path "data/test")) {
    New-Item -ItemType Directory -Force -Path data/fer2013 | Out-Null
    Move-Item data/train data/fer2013/
    Move-Item data/test data/fer2013/
}
```

# 8. Якщо всередині data/fer2013 ще одна папка fer2013 (типовий кейс деяких архівів), пересунемо все всередину "правильно"

### macOS / Linux:
```bash
if [ -d "data/fer2013/fer2013" ]; then
    mv data/fer2013/fer2013/train data/fer2013/
    mv data/fer2013/fer2013/test  data/fer2013/
    rm -rf data/fer2013/fer2013
fi
```

### Windows:
```powershell
if (Test-Path "data/fer2013/fer2013") {
    Move-Item data/fer2013/fer2013/train data/fer2013/
    Move-Item data/fer2013/fer2013/test  data/fer2013/
    Remove-Item -Recurse -Force data/fer2013/fer2013
}
```


# 9. Створюємо validation з потрібними класами

### macOS / Linux:
```bash
mkdir -p data/fer2013/validation/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}
```

### Windows:
```powershell
$classes = "Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"
foreach ($d in $classes) {
    New-Item -ItemType Directory -Force -Path "data/fer2013/validation/$d" | Out-Null
}
```

# 10. Переносимо по 100 зображень з train у validation для кожного класу

### macOS / Linux:
```bash
for d in Angry Disgust Fear Happy Neutral Sad Surprise; do
    ls "data/fer2013/train/$d" | head -n 100 | while read f; do
        mv "data/fer2013/train/$d/$f" "data/fer2013/validation/$d/"
    done
done
```

### Windows:
```powershell
$classes = "Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"
foreach ($d in $classes) {
    Get-ChildItem "data/fer2013/train/$d" | Select-Object -First 100 |
        ForEach-Object {
            Move-Item $_.FullName "data/fer2013/validation/$d/"
        }
}
```

Після цього має з’явитися структура:
data/
└── fer2013/
    ├── train/
    ├── validation/
    └── test/


# 11. Запуск моделей: 
обовʼязково знаходьтесь в корені проєкту: 

- Простий рівень: 
### Mac / Linux / Windows
```bash
 python train_fer.py --model simple --data_root data/fer2013
```
- Середній: 
### Mac / Linux / Windows
```bash
python train_fer.py --model cnn --data_root data/fer2013
```
- Складний: 
### Mac / Linux / Windows
```bash
python train_fer.py --model vit --data_root data/fer2013
```

Дочекайтесь закінчення кожної з них, проста пробіжить швидко , в середній 10 епох, в складній 1 епоха прописана

Можете для зручності запустити в трьох різних терміналах, і змінити кількість епох 

# 12. Якщо хочете протестувати на зображенні виконайте: 
### Mac / Linux
```bash
python -c "from train_fer import predict_image_cnn; predict_image_cnn('data/fer2013/test/Happy/PrivateTest_647018.jpg')" 
```
   - але замість Happy/PrivateTest_647018.jpg , напишіть шлях до того яке хочете протестувати

### Windows
```powershell
python -c "from train_fer import predict_image_cnn; predict_image_cnn('data/fer2013/test/Happy/PrivateTest_647018.jpg')"
```
