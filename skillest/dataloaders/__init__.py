ACTITRCKER_DIR = "activities/Actitracker"
ACTITRCKER_XML_FILEPATH = "activities/Actitracker/Actitracker.xml"
ACTITRCKER_SAMPLE_RATE_PER_SEC = 20
ACTITRCKER_ACTIVITIES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
ACTITRCKER_ACTIVITIES_TO_IDX = {a: i for i, a in enumerate(ACTITRCKER_ACTIVITIES)}


from skillest.dataloaders.actitracker_dl import ActitrackerDL
from skillest.dataloaders.imu_dl import IMUDataModule, IMUDataset