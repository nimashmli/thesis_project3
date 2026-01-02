import os
import json
from pathlib import Path
from datetime import datetime


class ExperimentManager:
    """
    مدیریت پوشه‌ها و فایل‌های JSON برای ذخیره نتایج آزمایش‌ها
    """
    
    def __init__(self, base_dir="/content/drive/MyDrive/result"):
        """
        Parameters:
        -----------
        base_dir : str
            مسیر پایه در Google Drive (پیش‌فرض: /content/drive/MyDrive/result)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_run_path(self, model_name, validation_type, run_number=None):
        """
        ساخت مسیر پوشه run
        
        Parameters:
        -----------
        model_name : str
            نام مدل
        validation_type : str
            subject_independent یا subject_dependent
        run_number : int, optional
            شماره run (اگر None باشد، شماره بعدی را پیدا می‌کند)
        
        Returns:
        --------
        Path : مسیر پوشه run
        int : شماره run
        """
        model_dir = self.base_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        validation_dir = model_dir / validation_type
        validation_dir.mkdir(exist_ok=True)
        
        if run_number is None:
            # پیدا کردن شماره run بعدی
            existing_runs = [d.name for d in validation_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('run')]
            if existing_runs:
                run_numbers = [int(r.replace('run', '')) for r in existing_runs 
                              if r.replace('run', '').isdigit()]
                run_number = max(run_numbers) + 1 if run_numbers else 1
            else:
                run_number = 1
        
        run_dir = validation_dir / f"run{run_number}"
        run_dir.mkdir(exist_ok=True)
        
        return run_dir, run_number
    
    def create_experiment_config(self, run_dir, model_name, emotion, category, k, 
                                 validation_type, hyperparameters, total_epochs):
        """
        ایجاد فایل JSON برای ذخیره اطلاعات آزمایش
        
        Parameters:
        -----------
        run_dir : Path
            مسیر پوشه run
        model_name : str
            نام مدل
        emotion : str
            نوع احساس
        category : str
            دسته‌بندی
        k : int
            تعداد fold
        validation_type : str
            نوع اعتبارسنجی
        hyperparameters : dict
            هایپرپارامترها
        total_epochs : int
            تعداد کل epochs
        
        Returns:
        --------
        Path : مسیر فایل JSON
        """
        config_path = run_dir / "experiment_config.json"
        
        config = {
            "model_name": model_name,
            "emotion": emotion,
            "category": category,
            "k": k,
            "validation_type": validation_type,
            "hyperparameters": hyperparameters,
            "total_epochs": total_epochs,
            "current_epoch": 0,
            "completed_epochs": 0,
            "status": "in_progress",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "checkpoint_path": str(run_dir / "checkpoint.pth"),
            "log_path": str(run_dir / "training_log.json"),
            "last_completed_subject": -1,  # برای subject_dependent: -1 یعنی هنوز شروع نشده
            "last_completed_fold": -1,  # برای subject_dependent: آخرین fold پردازش شده در subject فعلی
            "current_subject": -1,  # برای subject_dependent: subject فعلی که در حال پردازش است
            "accuracies": {"train": [], "test": []}  # نتایج قبلی برای subject_dependent
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        return config_path
    
    def load_experiment_config(self, config_path):
        """
        بارگذاری فایل JSON آزمایش
        
        Parameters:
        -----------
        config_path : str or Path
            مسیر فایل JSON
        
        Returns:
        --------
        dict : اطلاعات آزمایش
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    def update_experiment_config(self, config_path, **updates):
        """
        به‌روزرسانی فایل JSON آزمایش
        
        Parameters:
        -----------
        config_path : str or Path
            مسیر فایل JSON
        **updates : dict
            فیلدهای برای به‌روزرسانی
        """
        config = self.load_experiment_config(config_path)
        config.update(updates)
        config["updated_at"] = datetime.now().isoformat()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    def find_experiment_by_config(self, model_name, emotion, category, k, 
                                  validation_type, hyperparameters):
        """
        پیدا کردن آزمایش موجود بر اساس پارامترها
        
        Parameters:
        -----------
        model_name : str
        emotion : str
        category : str
        k : int
        validation_type : str
        hyperparameters : dict
        
        Returns:
        --------
        Path or None : مسیر فایل config اگر پیدا شود، None در غیر این صورت
        """
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            return None
        
        validation_dir = model_dir / validation_type
        if not validation_dir.exists():
            return None
        
        # جستجو در همه runها
        for run_dir in validation_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith('run'):
                config_path = run_dir / "experiment_config.json"
                if config_path.exists():
                    config = self.load_experiment_config(config_path)
                    # بررسی تطابق پارامترها
                    if (config["model_name"] == model_name and
                        config["emotion"] == emotion and
                        config["category"] == category and
                        config["k"] == k and
                        config["validation_type"] == validation_type and
                        config["hyperparameters"] == hyperparameters and
                        config["status"] == "in_progress"):
                        return config_path
        
        return None

