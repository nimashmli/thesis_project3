from model_use.simpleNN import (
    create_model as simpleNN,
    subject_dependent_validation as simpleNN_sub_dep,
)
from model_use.cnn_45138 import (
    create_model as cnn_45138,
    subject_dependent_validation as cnn_45138_sub_dep,
)
from model_use.capsnet2020 import (
    create_model as capsnet2020,
    subject_dependent_validation as capsnet2020_sub_dep,
)
from model_use.hippoLegS1 import (
    create_model as hippoLegS1,
    subject_dependent_validation as hippoLegS1_sub_dep,
)

# یک رجیستری ساده برای نگهداری توابع مدل‌ها
MODEL_REGISTRY = {
    "simpleNN": (simpleNN, simpleNN_sub_dep),
    "cnn_45138": (cnn_45138, cnn_45138_sub_dep),
    "capsnet2020": (capsnet2020, capsnet2020_sub_dep),
    "hippoLegS1": (hippoLegS1, hippoLegS1_sub_dep),
}


def get_model_fns(name):
    """
    دریافت توابع حالت مستقل/وابسته به سوژه برای یک مدل.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered.")
    indep_fn, dep_fn = MODEL_REGISTRY[name]
    return indep_fn, dep_fn


def choose_model(
    name,
    emotion,
    category,
    test_person=None,
    fold_idx=None,
    subject_dependecy="subject_independent",
    **kwargs,
):
    """
    انتخاب و اجرای مدل بر اساس حالت subject-independent یا subject-dependent.
    """
    indep_fn, dep_fn = get_model_fns(name)

    if subject_dependecy == "subject_independent":
        return indep_fn(test_person, emotion, category, fold_idx, **kwargs)
    if subject_dependecy == "subject_dependent":
        return dep_fn(emotion, category, fold_idx, **kwargs)

    raise ValueError("subject_dependecy must be 'subject_independent' or 'subject_dependent'")
