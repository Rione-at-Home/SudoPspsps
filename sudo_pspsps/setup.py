from setuptools import find_packages, setup

package_name = 'sudo_pspsps'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ri-one',
    maintainer_email='gilsocojp@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'head_node = sudo_pspsps.head_node:main',
        'camera_node = sudo_pspsps.camera_node:main',
        'brain_node = sudo_pspsps.brain_node:main',
        'lfm_bridge_node = sudo_pspsps.lfm_bridge_node:main',
        'stt_node = sudo_pspsps.stt_node:main',
        'tts_node = sudo_pspsps.tts_node:main',
        'cat_brain_planner = sudo_pspsps.cat_brain_planner:main',
        'person_targeting_node = sudo_pspsps.person_targetting_node:main'
    ],
},
)
