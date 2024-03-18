Fog Detection and Alerting System in ADAS

The Fog Detection and Alerting System in ADAS (Advanced Driver Assistance Systems) is a project aimed at enhancing road safety by detecting foggy conditions and alerting drivers accordingly. The system employs two distinct approaches for fog detection: Support Vector Machine (SVM) classifier and VGG16 model.
Approaches:

    Support Vector Machine (SVM) Classifier:
        The SVM classifier is trained to analyze various features extracted from images to determine the presence of fog.
        Based on the output of the SVM classifier, the system activates fog lights (LEDs) connected to an Arduino board to alert drivers about the foggy conditions.

    VGG16 Model:
        The VGG16 model, a convolutional neural network (CNN) architecture, is utilized for fog detection by leveraging its deep learning capabilities.
        Similar to the SVM approach, the VGG16 model analyzes images to detect fog and triggers the fog lights via Arduino based on its output.

Implementation:

    The system integrates the output from both the SVM classifier and VGG16 model to enhance the accuracy and reliability of fog detection.
    Upon detecting foggy conditions, the system activates the fog lights connected to Arduino, providing visual alerts to drivers and improving visibility on the road.

How it Works:

    Image Processing:
        Input images captured by onboard cameras are processed using both SVM classifier and VGG16 model to detect fog.

    Decision Making:
        The outputs from both approaches are combined to make a decision regarding the presence of fog.

    Alerting System:
        If fog is detected, the system triggers the fog lights (LEDs) via Arduino, alerting drivers about the hazardous conditions.

Future Enhancements:

    Continuous refinement and optimization of fog detection algorithms to improve accuracy and reliability.
    Integration with other ADAS features for a comprehensive road safety solution.

Credits:

This project was developed by [Your Name/Organization]. Special thanks to the contributors for their efforts in implementing the SVM classifier, VGG16 model, and Arduino integration.
