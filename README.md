# CIFAR-10 Görüntü Sınıflandırma Kullanarak Transfer Learning
Bu proje, CIFAR-10 veri kümesindeki görüntülerin sınıflandırılması için transfer learning yöntemini kullanılmıştır. CIFAR-10 veri kümesi, 10 farklı sınıfa ait renkli (32x32 piksel) görüntülerden oluşur: uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi ve kamyon.

## Transfer Learning Kullanımı
Projede, önceden eğitilmiş derin öğrenme modelleri (DenseNet121, VGG16 ve ResNet50) kullanılarak transfer learning yöntemi uygulanmıştır. Bu modellerin önceden eğitilmiş ağırlıkları, ImageNet gibi büyük ve genel bir veri kümesi üzerinde eğitilerek elde edilmiştir. CIFAR-10 veri kümesinde ise daha küçük boyutlu ve özgün sınıflar bulunmaktadır. Transfer learning, bu tür önceden eğitilmiş ağırlıkları alıp, yeni bir görev için uygun hale getirerek eğitim sürecini hızlandırır ve performansı artırır. Bu projede, önceden eğitilmiş modellerin temelini oluşturarak, son katmanlarını CIFAR-10 veri kümesine göre özelleştirme yaklaşımı kullanılmıştır.

## Gereksinimler

Python 

TensorFlow 

## Katkıda Bulunma
Her türlü katkı ve geri bildirim için açığız. Lütfen bir GitHub issue oluşturun veya bir pull request gönderin.
