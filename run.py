from kfold_validation import validate
from model_use.main import choose_model
from plot import plot_training_history, plot_subject_dependet
from experiment_manager import ExperimentManager
import sys
import numpy as np
import json
from pathlib import Path


def extract_hyperparameters_from_model(model_name, emotion, category, validation_type='subject_independent'):
    """
    استخراج هایپرپارامترها بر اساس مدل
    """
    if model_name == 'simpleNN':
        if validation_type == 'subject_independent':
            return {
                "overlap": 0.0,
                "time_len": 1,
                "batch_size": 128,
                "lr": 1e-3,
                "epochs": 30,
                "optimizer": "Adam"
            }
        else:  # subject_dependent
            return {
                "overlap": 0.05,
                "time_len": 1,
                "batch_size": 128,
                "lr": 1e-3,
                "epochs": 30,
                "optimizer": "Adam"
            }
    
    elif model_name == 'cnn_45138':
        if validation_type == 'subject_independent':
            return {
                "overlap": 0.2,
                "time_len": 3,
                "batch_size": 250,
                "lr": 7e-4,
                "epochs": 20,
                "optimizer": "Adam"
            }
        else:  # subject_dependent
            return {
                "overlap": 0.05,
                "time_len": 1,
                "batch_size": 64,
                "lr": 2e-4,
                "epochs": 15,
                "optimizer": "Adam"
            }
    
    elif model_name == 'capsnet2020':
        if validation_type == 'subject_independent':
            return {
                "overlap": 0.0,
                "time_len": 1,
                "batch_size": 256,
                "lr": 2e-5,
                "epochs": 30,
                "optimizer": "Adam",
                "num_filter": 256,
                "caps_len": 8,
                "out_dim": 16
            }
        else:  # subject_dependent
            return {
                "overlap": 0.0,
                "time_len": 1,
                "batch_size": 128,
                "lr": 1e-4,
                "epochs": 30,
                "optimizer": "Adam",
                "num_filter": 32,
                "caps_len": 4,
                "out_dim": 8
            }
    
    elif model_name == 'hippoLegS1':
        if validation_type == 'subject_independent':
            return {
                "overlap": 0.0,
                "time_len": 1,
                "batch_size": 64,
                "lr": 5e-5,
                "epochs": 25,
                "optimizer": "Adam",
                "x_dim": 14,
                "h_dim": 24,
                "c_dim": 64,
                "dim2": 64,
                "dim3": 16
            }
        else:  # subject_dependent
            return {
                "overlap": 0.0,
                "time_len": 2,
                "batch_size": 64,
                "lr": 5e-5,
                "epochs": 30,
                "optimizer": "Adam",
                "x_dim": 14,
                "h_dim": 32,
                "c_dim": 32,
                "dim2": 64,
                "dim3": 16
            }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name=None, emotion=None, category=None, k=None, 
                  validation_type=None, num_people=23, config_path=None):
    """
    اجرای آزمایش با دو حالت: شروع جدید یا ادامه از checkpoint
    
    Parameters:
    -----------
    model_name : str, optional
        نام مدل (برای حالت جدید)
    emotion : str, optional
        نوع احساس (برای حالت جدید)
    category : str, optional
        دسته‌بندی (برای حالت جدید)
    k : int, optional
        تعداد fold (برای حالت جدید)
    validation_type : str, optional
        نوع اعتبارسنجی (برای حالت جدید)
    num_people : int
        تعداد افراد
    config_path : str, optional
        مسیر فایل JSON برای ادامه از checkpoint
    
    Returns:
    --------
    dict : نتایج آزمایش
    """
    manager = ExperimentManager()
    
    # حالت ادامه از checkpoint
    if config_path:
        print("="*60)
        print("🔄 Resuming from previous experiment...")
        print("="*60)
        
        config = manager.load_experiment_config(config_path)
        run_dir = Path(config_path).parent
        
        print(f"\n📁 Experiment Directory: {run_dir}")
        print(f"📊 Model: {config['model_name']}")
        print(f"📈 Emotion: {config['emotion']}")
        print(f"🏷️  Category: {config['category']}")
        print(f"🔢 K-folds: {config['k']}")
        print(f"📋 Validation Type: {config['validation_type']}")
        print(f"\n📉 Previous Progress:")
        if config['validation_type'] == 'subject_dependent':
            last_subject = config.get('last_completed_subject', -1)
            completed_subjects = len(config.get('accuracies', {}).get('train', []))
            print(f"   - Completed Subjects: {completed_subjects}/23")
            print(f"   - Last Completed Subject: {last_subject}")
            print(f"   - Remaining Subjects: {23 - completed_subjects}")
            if completed_subjects > 0:
                prev_accuracies = config.get('accuracies', {})
                if prev_accuracies.get('train') and prev_accuracies.get('test'):
                    print(f"   - Previous Results:")
                    print(f"     * Average Train Acc: {np.mean(prev_accuracies['train']):.2f}%")
                    print(f"     * Average Test Acc: {np.mean(prev_accuracies['test']):.2f}%")
        else:
            print(f"   - Completed Epochs: {config['completed_epochs']}")
            print(f"   - Total Epochs: {config['total_epochs']}")
            print(f"   - Remaining Epochs: {config['total_epochs'] - config['completed_epochs']}")
        print(f"   - Status: {config['status']}")
        if config['validation_type'] == 'subject_dependent':
            last_subject = config.get('last_completed_subject', -1)
            print(f"\n🔄 Resuming from Subject {last_subject + 1}...")
        else:
            print(f"\n🔄 Resuming training from epoch {config['completed_epochs'] + 1}...")
        print("="*60)
        
        # استخراج پارامترها از config
        model_name = config['model_name']
        emotion = config['emotion']
        category = config['category']
        k = config['k']
        validation_type = config['validation_type']
        hyperparameters = config['hyperparameters']
        
        # ادامه آموزش
        if validation_type == 'subject_independent':
            train_loss, val_loss, train_acc, val_acc = validate(
                model_name, emotion, category, k, num_people, 
                run_dir=run_dir, config=config, resume=True
            )
            
            history = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            
            # ذخیره نتایج
            results_path = run_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            # رسم نمودارها
            plot_training_history(history, save_dir=run_dir)
            
            # به‌روزرسانی config
            manager.update_experiment_config(
                config_path,
                status="completed",
                completed_epochs=config['total_epochs'],
                current_epoch=config['total_epochs']
            )
            
            print(f"\n=== Final Results (averaged over {k} folds) ===")
            print(f"Average Train Loss: {np.mean(train_loss[-5:]):.4f}")
            print(f"Average Val Loss: {np.mean(val_loss[-5:]):.4f}")
            print(f"Average Train Accuracy: {np.mean(train_acc[-5:]):.2f}%")
            print(f"Average Val Accuracy: {np.mean(val_acc[-5:]):.2f}%")
            
            return history
        
        else:  # subject_dependent
            accuracies = choose_model(
                model_name, emotion, category, None, None,
                subject_dependecy='subject_dependent',
                run_dir=run_dir, config=config, resume=True
            )
            
            # محاسبه میانگین و واریانس (تعداد واقعی subject‌های پردازش شده)
            completed_count = len(accuracies['test'])
            if completed_count > 0:
                avg_test_acc = np.sum(accuracies['test']) / completed_count
                avg_train_acc = np.sum(accuracies['train']) / completed_count
            else:
                avg_test_acc = 0
                avg_train_acc = 0
            
            _test_accs = np.array(accuracies['test'], dtype=float)
            _train_accs = np.array(accuracies['train'], dtype=float)
            var_test_acc = np.var(_test_accs, ddof=1)
            var_train_acc = np.var(_train_accs, ddof=1)
            
            # ذخیره نتایج
            results = {
                'test': accuracies['test'].tolist() if isinstance(accuracies['test'], np.ndarray) else accuracies['test'],
                'train': accuracies['train'].tolist() if isinstance(accuracies['train'], np.ndarray) else accuracies['train'],
                'avg_test_acc': float(avg_test_acc),
                'avg_train_acc': float(avg_train_acc),
                'var_test_acc': float(var_test_acc),
                'var_train_acc': float(var_train_acc)
            }
            
            results_path = run_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            # به‌روزرسانی config
            manager.update_experiment_config(
                config_path,
                status="completed",
                completed_epochs=config['total_epochs'],
                current_epoch=config['total_epochs']
            )
            
            print(f"\n=== Final Results (averaged over {completed_count} subjects) ===")
            print(f"Average Test Accuracy: {avg_test_acc:.2f}%")
            print(f"Average Train Accuracy: {avg_train_acc:.2f}%")
            if completed_count > 1:
                print(f"Variance Test Accuracy: {var_test_acc:.6f}")
                print(f"Variance Train Accuracy: {var_train_acc:.6f}")
            
            # رسم نمودار
            plot_subject_dependet(accuracies, save_dir=run_dir)
            
            return results
    
    # حالت شروع جدید
    if not all([model_name, emotion, category, k, validation_type]):
        raise ValueError("All parameters must be provided for new experiment")
    
    print("="*60)
    print("🚀 Starting new experiment...")
    print("="*60)
    
    # استخراج هایپرپارامترها
    hyperparameters = extract_hyperparameters_from_model(model_name, emotion, category, validation_type)
    
    # پیدا کردن run موجود با همین پارامترها
    existing_config = manager.find_experiment_by_config(
        model_name, emotion, category, k, validation_type, hyperparameters
    )
    
    if existing_config:
        print(f"\n⚠️  Found existing in-progress experiment at: {existing_config}")
        print("   Use config path to resume instead of starting new one.")
        response = input("   Continue with new run? (y/n): ")
        if response.lower() != 'y':
            return None
        run_dir, run_number = manager.get_run_path(model_name, validation_type)
    else:
        run_dir, run_number = manager.get_run_path(model_name, validation_type)
    
    print(f"\n📁 Experiment Directory: {run_dir}")
    print(f"🔢 Run Number: {run_number}")
    
    # ایجاد فایل config
    total_epochs = hyperparameters.get('epochs', 30)
    config_path = manager.create_experiment_config(
        run_dir, model_name, emotion, category, k, validation_type,
        hyperparameters, total_epochs
    )
    
    print(f"✅ Experiment config created: {config_path}")
    print(f"📋 Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    
    # اجرای آزمایش
    if validation_type == 'subject_independent':
        print(f"\n🚀 Running Subject-Independent validation with {k}-fold cross-validation...")
        
        train_loss, val_loss, train_acc, val_acc = validate(
            model_name, emotion, category, k, num_people, run_dir=run_dir, config_path=config_path
        )
        
        history = {
            'train_loss': train_loss.tolist() if isinstance(train_loss, np.ndarray) else train_loss,
            'val_loss': val_loss.tolist() if isinstance(val_loss, np.ndarray) else val_loss,
            'train_acc': train_acc.tolist() if isinstance(train_acc, np.ndarray) else train_acc,
            'val_acc': val_acc.tolist() if isinstance(val_acc, np.ndarray) else val_acc
        }
        
        # ذخیره نتایج
        results_path = run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # رسم نمودارها
        plot_training_history(history, save_dir=run_dir)
        
        # به‌روزرسانی config
        manager.update_experiment_config(
            config_path,
            status="completed",
            completed_epochs=total_epochs,
            current_epoch=total_epochs
        )
        
        print(f"\n=== Final Results (averaged over {k} folds) ===")
        print(f"Average Train Loss: {np.mean(train_loss[-5:]):.4f}")
        print(f"Average Val Loss: {np.mean(val_loss[-5:]):.4f}")
        print(f"Average Train Accuracy: {np.mean(train_acc[-5:]):.2f}%")
        print(f"Average Val Accuracy: {np.mean(val_acc[-5:]):.2f}%")
        
        return history
        
    elif validation_type == 'subject_dependent':
        print(f"\n🚀 Running Subject-Dependent validation with {k}-fold cross-validation per subject...")
        
        accuracies = choose_model(
            model_name, emotion, category, None, None,
            subject_dependecy='subject_dependent',
            run_dir=run_dir, config_path=config_path
        )
        
        # محاسبه میانگین و واریانس
        avg_test_acc = np.sum(accuracies['test']) / num_people
        avg_train_acc = np.sum(accuracies['train']) / num_people
        
        _test_accs = np.array(accuracies['test'], dtype=float)
        _train_accs = np.array(accuracies['train'], dtype=float)
        var_test_acc = np.var(_test_accs, ddof=1)
        var_train_acc = np.var(_train_accs, ddof=1)
        
        # ذخیره نتایج
        results = {
            'test': accuracies['test'].tolist() if isinstance(accuracies['test'], np.ndarray) else accuracies['test'],
            'train': accuracies['train'].tolist() if isinstance(accuracies['train'], np.ndarray) else accuracies['train'],
            'avg_test_acc': float(avg_test_acc),
            'avg_train_acc': float(avg_train_acc),
            'var_test_acc': float(var_test_acc),
            'var_train_acc': float(var_train_acc)
        }
        
        results_path = run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # به‌روزرسانی config
        manager.update_experiment_config(
            config_path,
            status="completed",
            completed_epochs=total_epochs,
            current_epoch=total_epochs
        )
        
        print(f"\n=== Final Results (averaged over {num_people} subjects) ===")
        print(f"Average Test Accuracy: {avg_test_acc:.2f}%")
        print(f"Average Train Accuracy: {avg_train_acc:.2f}%")
        print(f"Variance Test Accuracy: {var_test_acc:.6f}")
        print(f"Variance Train Accuracy: {var_train_acc:.6f}")
        
        # رسم نمودار
        plot_subject_dependet(accuracies, save_dir=run_dir)
        
        return results
    
    else:
        raise ValueError(
            f"Invalid validation_type: {validation_type}. "
            "Must be 'subject_independent' or 'subject_dependent'"
        )


def main():
    """
    تابع اصلی برای اجرا از command line
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  New experiment:")
        print("    python run.py <model_name> <emotion> <category> <k> <validation_type>")
        print("  Resume experiment:")
        print("    python run.py <config_json_path>")
        print("\nParameters for new experiment:")
        print("  model_name      : simpleNN, cnn_45138, capsnet2020, hippoLegS1")
        print("  emotion         : valence, dominance")
        print("  category        : binary, 5category")
        print("  k               : number of folds (integer)")
        print("  validation_type : subject_independent, subject_dependent")
        print("\nExample:")
        print("  python run.py simpleNN valence binary 5 subject_independent")
        print("  python run.py /content/drive/MyDrive/result/cnn_45138/subject_dependent/run1/experiment_config.json")
        sys.exit(1)
    
    # بررسی اینکه آیا اولین آرگومان یک مسیر JSON است
    first_arg = sys.argv[1]
    if first_arg.endswith('.json') and Path(first_arg).exists():
        # حالت ادامه از checkpoint
        config_path = first_arg
        results = run_experiment(config_path=config_path)
    else:
        # حالت شروع جدید
        if len(sys.argv) < 6:
            print("Error: Not enough arguments for new experiment")
            sys.exit(1)
        
        model_name = sys.argv[1]
        emotion = sys.argv[2]
        category = sys.argv[3]
        k = int(sys.argv[4])
        validation_type = sys.argv[5]
        
        results = run_experiment(
            model_name=model_name,
            emotion=emotion,
            category=category,
            k=k,
            validation_type=validation_type
        )
    
    return results


if __name__ == "__main__":
    main()
