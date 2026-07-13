import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pic4people_tracking'

def recursive_glob(folder):
    return [f for f in glob(os.path.join(folder, '**', '*'), recursive=True) if os.path.isfile(f)]

models_dir = os.path.join(os.path.dirname(__file__), 'models')
model_files = recursive_glob(models_dir)

model_install_files = [
    (os.path.join('share', package_name, 'models', os.path.relpath(os.path.dirname(f), models_dir)), [f])
    for f in model_files
]
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params', '*.yaml'))),
    ] + model_install_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='eiraleandrea@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker = pic4people_tracking.tracker:main',
            'tracker_fast = pic4people_tracking.tracker_fast:main',
            'to_csv_node = pic4people_tracking.to_csv_node:main',
            'test_image_format = pic4people_tracking.test_image_format:main',
            'fusion_node = pic4people_tracking.fusion_node:main',
            'tracker_ReID = pic4people_tracking.tracker_ReID:main',
            'fusion_node_ReID = pic4people_tracking.fusion_node_ReID:main',
        ],
    },
)
