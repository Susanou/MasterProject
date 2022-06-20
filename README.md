# MasterProject

My Master Thesis project on GAN attacks and mitigations to it.
Algorithm POC and privacy measurement

## [Report](MS_project_report_clh137.pdf)

Previous research has shown that federated learning is vulnerable to multiple
attacks, in particular whitebox attacks. Even limiting the number of parameters
shared can still lead to the training data of the victim being leaked. This is problematic
especially when considering laws like GDPR and HIPAA that demand that
sensitive data be protected. Our alternative proposes to only share the output labels
themselves of the training phase to a server who would then choose the correct
label and send it back to all participants. This ensures that only blackbox attacks
could be performed on the system. Our preliminary results seem to show that this
would make it much harder to run membership inference attacks on this system.

![privacy analysis](images/report.png)