import imp
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import altair as alt
import joblib



st.title('Aplikasi Web Data Mining')
st.write("""
    Nama : Saiyidati Vienna Arum Pratama \n
    Nim  : 200411100018 \n
    Kelas: Penambangan Data IF 5A \n
    """
    )


menu = st.sidebar.selectbox(
    'Pilihan Menu',
    ('Home','Dataset','Preprocessing', 'Modelling', 'Implementasi')
)

if menu == 'Home':
    st.write("""
    # Data Mining
    Data mining adalah proses pengumpulan dan pengolahan data yang bertujuan untuk mengekstrak informasi penting pada data. Proses pengumpulan dan ekstraksi informasi tersebut dapat dilakukan menggunakan perangkat lunak dengan bantuan perhitungan statistika, matematika, ataupun teknologi Artificial Intelligence (AI). Data mining sering disebut juga Knowledge Discovery in Database (KDD).
    
    Tujuan data mining Data mining dilakukan untuk memenuhi beberapa tujuan tertentu, antara lain :
    
    1. Sebagai sarana menjelaskan (Explanatory) Data mining dapat digunakan sebagai sarana untuk menjelaskan suatu kondisi penelitian.
    2. Sebagai sarana konfirmasi (Confirmatory) Data mining dapat digunakan sebagai sarana untuk memastikan sebuah pernyataan atau mempertegas suatu hipotesis.
    3. Sebagai sarana eksplorasi (Exploratory) Data mining dapat digunakan sebagai sarana untuk mencari pola baru yang sebelumnya tidak terdeteksi. Metode data mining.
    Metode yang digunakan untuk melakukan data mining, sebagai berikut :
    1. Association Teknik yang pertama adalah association. Association adalah metode berbasis aturan yang digunakan untuk menemukan asosiasi dan hubungan variabel dalam satu set data. Biasanya analisis ini terdiri dari pernyataan “if atau then” sederhana. Association banyak digunakan dalam mengidentifikasi korelasi produk dalam keranjang belanja untuk memahami kebiasaan konsumsi pelanggan. Sehingga, perusahaan dapat mengembangkan strategi penjualan dan membuat sistem rekomendasi yang lebih baik.
    2. Classification Selanjutnya classification, ia adalah metode yang paling umum digunakan dalam data mining. Classification adalah tindakan untuk memprediksi kelas suatu objek.
    3. Regression Regression adalah teknik yang menjelaskan variabel dependen melalui proses analisis variabel independen. Sebagai contoh, prediksi penjualan suatu produk berdasarkan korelasi antara harga produk dengan tingkat pendapatan rata-rata pelanggan.
    4. Clustering Terakhir, metode clustering. Clustering digunakan dalam membagi kumpulan data menjadi beberapa kelompok berdasarkan kemiripan atribut yang dimiliki. Contoh kasusnya adalah Customer Segmentation. Ia membagi pelanggan ke dalam beberapa grup berdasarkan tingkat kemiripannya.\n
    """)

elif menu == 'Dataset' :
    st.title('Dataset')
    st.write("""
    Dataset yang digunakan di ambil dari kaggle, Berikut Alamat URL Dataset : \n
    https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?select=heart_cleveland_upload.csv
    """)
    # st.write("""#### Deskripsi Dataset""")
    # st.write("""
    # Dalam kumpulan data ini, 5 kumpulan data jantung digabungkan dengan 11 fitur umum yang menjadikannya kumpulan data penyakit jantung terbesar yang tersedia sejauh ini untuk tujuan penelitian. 
    # """
    # )
    st.write(
        """##### Dataset Prediksi Penyakit Jantung """)
    
    st.write("""
        Penyakit jantung dan pembuluh darah atau penyakit kardiovaskular adalah berbagai kondisi di mana terjadi penyempitan atau penyumbatan pembuluh darah yang dapat menyebabkan serangan jantung, nyeri dada (angina), atau stroke.
        Penyakit kardiovaskuler termasuk kondisi kritis yang butuh penanganan segera. Pasalnya, jantung adalah organ vital yang berfungsi untuk memompa darah ke seluruh tubuh. Jika jantung bermasalah, peredaran darah dalam tubuh bisa terganggu.
        """
    )
    st.write(
        """###### Atribut / Fitur Dataset Prediksi Penyakit Jantung : """)
    
    st.write("""
        Dalam Dataset ini terdapat kumpulan data berisi 13 fitur yang dapat digunakan untuk memprediksi kemungkinan penyakit jantung, sebagai berikut :
        1. Age : Berisi umur dari pasien (tahun)
        2. Sex : Berisi jenis kelamin dari pasien, dimana :\n
            1 = Laki - laki, dan \n
            0 = Perempuan
        3. Cp : Berisi tipe nyeri dada, dimana \n
            -- Nilai 0: angina tipikal \n
            -- Nilai 1: angina atipikal \n
            -- Nilai 2: nyeri non-angina \n
            -- Nilai 3: asimtomatik 
        4. Trestbps: yaitu tekanan darah (dalam mm Hg saat masuk rumah sakit)
        5. Chol : yaitu berisi kadar kolesterol serum [mg/dl]
        6. Fbs : yaitu gula darah > 120 mg/dl, dimana : \n
            1 = benar, dan \n
            0 = salah 
        7. Restecg : hasil elektrokardiografi, dimana : \n
            -- Nilai 0: normal \n
            -- Nilai 1: memiliki kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0,05 mV) \n
            -- Nilai 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes
        8. Thalach: Detak jantung maksimum tercapai
        9. Exang : angina akibat olahraga, dimana : \n
            1 = ya, dan \n
            0 = tidak
        10. Oldpeak : ST nilai numerik di ukur dalam depresi yang diinduksi oleh latihan relatif terhadap istirahat
        11. Slope : kemiringan puncak latihan segmen ST, dimana : \n
            -- Nilai 0: upsloping \n
            -- Nilai 1: flat \n
            -- Nilai 2: downsloping
        12. Ca : jumlah kapal utama (0-3) diwarnai oleh flourosopy
        13. Thal : dimana : \n
            0 = normal \n
            1 = cacat tetap \n
            2 = cacat reversibel
        """
    )
    
    # uploaded_files = st.file_uploader("Upload Dataset dengan Format .csv", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     df = pd.read_csv(uploaded_file)
    #     st.write("Nama File Anda = ", uploaded_file.name)
    #     st.dataframe(df)
    # st.write('Jumlah Baris dan Kolom : ', df.shape)
    st.write(
        """##### Dataset Heart Disease """)
    df = pd.read_csv('https://raw.githubusercontent.com/saiyidativiennaarumpratama/datamining/main/heart_cleveland_upload.csv')
    st.dataframe(df)
    st.write('Jumlah Baris dan Kolom : ', df.shape)

elif menu == 'Preprocessing':
    st.title('Preprocessing')
    tab1 = st.tabs(["Normalisasi"])
    st.write(
        'Proses Scaling atau biasa dikenal dengan normalisasi. Data yang memiliki rentang yang cukup jauh satu sama lain maka data itu perlu di normalisasi. '
        'Normalisasi data membutuhkan nilai minimum dan maksimum. Nilai minimum yang biasa digunakan adalah 0, dan nilai maksimum adalah 1. Sehingga, data memiliki rentang 0 sampai 1'
    )
    st.write(
        """##### Rumus Scaling """)
    st.latex(r''' x'=\frac{x-x_{min}}{x_{max}-x_{min}} 
    ''')
    st.write(
        """##### Data Asli """)
    df = pd.read_csv('https://raw.githubusercontent.com/saiyidativiennaarumpratama/datamining/main/heart_cleveland_upload.csv')
    st.dataframe(df)
  
    #memisahkan fitur dengan label(condition)
    #X = data yang conditionnya tidak ada
    X = df.drop(columns='condition', axis=1)
    # Y = data yang condition saja
    Y = df['condition']
    st.write(
        """##### Data Tanpa Label """)
    X
    st.write(
        """##### Data Label """)
    Y
    st.write(
        """##### Target Label """)
    st.write(
        '0 = Tidak sakit, 1 = Sakit')
    dumies = pd.get_dummies(df.condition).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
    })
    st.write(labels)

    #Normalisasi Min Max
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(X)
    #memasukan fitur 
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(data_scaled, columns=features_names)
    st.write(
        """##### Hasil Normalisasi Data """)
    st.write(scaled_features)
    X_training, X_test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    Y_training_label, Y_test_label = train_test_split(Y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    st.write("Jumlah Data", X.shape)
    st.write("""##### Data Training""", X_training)
    st.write("Jumlah Data Training", X_training.shape)
    st.write("""##### Data Test""", X_test)
    st.write("Jumlah Data Test", X_test.shape)

elif menu == 'Modelling':
    st.title('Modelling')
    
    df = pd.read_csv('https://raw.githubusercontent.com/saiyidativiennaarumpratama/datamining/main/heart_cleveland_upload.csv')

    #memisahkan fitur dengan label(condition)
    #X = data yang conditionnya tidak ada
    X = df.drop(columns='condition', axis=1)
    # Y = data yang condition saja
    Y = df['condition']
    # # X= data tanpa label(labelnya di drop)
    # X
    # # Y=data label
    # Y
    #Normalisasi Min Max
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(X)
    #memasukan fitur 
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(data_scaled, columns=features_names)
    # st.write(
    #     """##### Hasil Normalisasi Data """)
    # Scaled_features = hasil dari normalisasi
    # st.write(scaled_features)
    X_training, X_test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    Y_training_label, Y_test_label = train_test_split(Y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    # st.write("Jumlah Data", X.shape)
    # st.write("""##### Data Training""", X_training)
    # st.write("Jumlah Data Training", X_training.shape)
    # st.write("""##### Data Training""", X_test)
    # st.write("Jumlah Data Test", X_test.shape)
    
    NB, KNN, DT, Grafik_Akurasi_Semuamodel = st.tabs(["Gaussian Naive Bayes", "K-Nearest Neighbor", "Decission Tree", "Grafik Akurasi Semua Model"])
    with NB :
        st.write(
            """##### Deskripsi Naive Bayes """)
        st.write('''
        Algoritma Naive Bayes adalah algoritma yang mempelajari probabilitas suatu objek dengan ciri-ciri tertentu yang termasuk dalam kelompok/kelas tertentu. 
        Naive Bayes berfungsi memprediksi probabilitas di masa depan berdasarkan pengalaman sebelumnya, sehingga dapat digunakan untuk pengambilan keputusan. 
        Gaussian Naive Bayes adalah tipe Naive Bayes yang mengikuti distribusi normal Gaussian dan mendukung data kontinu. 
        ''')

        gnbclassifier = GaussianNB()
        # data training ntuk pembelajarannya
        gnbclassifier.fit(X_training, Y_training_label)
        # Prediksi 
        Y_pred = gnbclassifier.predict(X_test)
        # hitung cm dari data testing terhadap data prediksi
        cm_gnb = confusion_matrix(Y_test_label, Y_pred)
        st.write('Hasil Confusion Matrix', cm_gnb)

        # Hitung Akurasi
        ac_gnb = round (accuracy_score(Y_test_label, Y_pred)*100)
        st.write("Akurasi Gaussian Naive Bayes (in %): ", ac_gnb)
        

    with KNN :
        st.write(
            """##### Deskripsi KNN """)
        st.write('''
        KNN (K-Nearest Neighbor) adalah algoritma supervised untuk melakukan klasifikasi terhadap objek dengan data pembelajaran yang jaraknya paling dekat dengan objek tersebut. Kasus khusus di mana klasifikasi diprediksikan berdasarkan data pembelajaran yang paling dekat (dengan kata lain, k = 1).
        ''')
        st.write(
            """##### Rumus untuk menghitung bobot kemiripan (similarity) """)
        st.write('''
        Rumus untuk menghitung bobot kemiripan (similarity) dengan Nearest Neighbor digunakan rumus Euclidean.
        ''')
#         image = Image.open('jarakecludian.png')
#         st.image(image, caption='Rumus Jarak Euclidean')
        st.write(
            """##### Tujuan Algoritma K-NN """)
        st.write('''
        Tujuan dari algoritma K-NN adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data.
        ''')
        st.write(
            """##### Algoritma K-NN """)
        st.write('''
        1. Menentukan Nilai K
        2. Menghitung jarak data uji(data baru yang akan di evaluasi) dengan data latih(data lama yang sudah ada)
        3. Mengurutkan data berdasarkan jarak terkecil ke terbesar
        4. Mengambil data sebanyak k terdekat, yaitu memiliki k terdekat (tetangga terdekat)
        5. Menentukan kelas dari data baru sesui dengan k terdekat
        ''')
        K = st.sidebar.slider('K',1, 30)
        knnclassifier = KNeighborsClassifier(K)
        # data training untuk pembelajarannya
        knnclassifier.fit(X_training, Y_training_label)
        # Prediksi
        Y_pred = knnclassifier.predict(X_test)

        cm_knn = confusion_matrix(Y_test_label, Y_pred)
        st.write('Hasil Confusion Matrix', cm_knn)
        ac_knn = round (accuracy_score(Y_test_label, Y_pred)*100)
        st.write("Akurasi KNN (in %): ", ac_knn)

    with DT :
        st.write(""" ##### Deskripsi Decission Tree
        """)
        st.write('''
        Decission Tree merupakan model analisis pemecahan masalah pengambilan keputusan, yang mana pemetaan mengenai alternatif-alternatif pemecahan masalah dapat diambil dari masalah tersebut. Pohon tersebut juga memperlihatkan faktor-faktor kemungkinan atau probablitas yang akan mempengaruhi alternatif keputusan disertai dengan estimasi hasil akhir yang akan didapat apabila mengambil alternatif keputusan tersebut.
        ''')
        st.write(""" ##### Jenis Pohon Keputusan berdasarkan variabel sasaran
        """)
        st.write('''
        Jenis pohon keputusan berdasarkan variabel sasaran, yaitu pohon keputusan variabel kategorikal dan pohon keputusan variabel kontinu.
        ''')
        st.write(""" ##### Pohon Keputusan Variabel Kategori
        """)
        st.write('''
        Pohon keputusan variabel kategori merupakan sebuah pohon keputusan variabel kategoris termasuk variabel target kategoris dibagi ke dalam kategori.
        ''')
        st.write(""" ##### Pohon Keputusan Variabel Kontinu
        """)
        st.write('''
        Pohon keputusan variabel kontinu adalah pohon keputusan dengan variabel target kontinu.
        ''')

        dtclassifier = DecisionTreeClassifier()
        dtclassifier.fit(X_training, Y_training_label)
        # presiksi
        Y_pred = dtclassifier.predict(X_test)

        cm_dt = confusion_matrix(Y_test_label, Y_pred)
        st.write("Hasil Confusion Matrix", cm_dt)
        ac_dt = round (accuracy_score(Y_test_label, Y_pred)*100)
        st.write("Akurasi DT (in %): ", ac_dt)


    with Grafik_Akurasi_Semuamodel:
        st.write ("##### Grafik Akurasi Semua Model")
        # grafik = st.button("Grafik AKurasi Semua Model")
        # if grafik : 
        data = pd.DataFrame({
            'Akurasi' : [ac_gnb, ac_knn, ac_dt],
            'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                alt.X("Akurasi"),
                alt.Y("Model"),
                alt.Color("Akurasi"),
                alt.Tooltip(["Akurasi", "Model"]),
            )
            .interactive()
            )
        st.altair_chart(chart,use_container_width=True, theme="streamlit")

        
        



elif menu == 'Implementasi':
    # Dataset
    df = pd.read_csv('https://raw.githubusercontent.com/saiyidativiennaarumpratama/datamining/main/heart_cleveland_upload.csv')

    # X = data yang tidak ada label
    X = df.drop(columns='condition', axis=1)
    # Y = data label
    Y = df['condition']


    #Normalisasi Min Max
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(X)
    #memasukan fitur 
    features_names = X.columns.copy()
    #features_names.remove('label')
    # Scaled_features = hasil dari normalisasi
    scaled_features = pd.DataFrame(data_scaled, columns=features_names)
    # save scaled 
    scaler_filename = "df_scaled.save"
    joblib.dump(scaler, scaler_filename)

    # Split Data
    X_training, X_test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    Y_training_label, Y_test_label = train_test_split(Y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("my_form"):
        st.subheader("implementasi")
        age = st.number_input('Masukkan Age (Umur)')

        sex = st.selectbox('Sex (Jenis Kelamin)', ('Perempuan', 'Laki - Laki'))
        if sex == 'Perempuan':
            sex = 0
        elif sex == 'Laki - Laki':
            sex = 1

        Cp = st.selectbox('Pilih Cp (Tipe Nyeri Dada)', ('Angina Tipikal', 'Angina Atipikal', 'Nyeri non-angina', 'Asimtomatik'))
        if Cp == 'Angina Tipikal':
            Cp = 0
        elif Cp == 'Angina Atipikal':
            Cp = 1
        elif Cp == 'Nyeri non-angina':
            Cp = 2
        elif Cp == 'Asimtomatik':
            Cp = 3

        trestbps = st.number_input('Masukkan Trestbps (Tekanan Darah (mm))')
        chol = st.number_input('Masukkan Chol (Kadar Kolesterol (mg/dl))')
        
        fbs = st.selectbox('Apakah Fbs (Gula Darah) > 120 mg/dl',('Salah','Benar'))
        if fbs == 'Salah':
            fbs = 0
        if fbs == 'Benar':
            fbs = 1

        restecg = st.selectbox('Restecg',('Normal','Memiliki kelainan gelombang ST-T', 'Kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes'))
        if restecg == 'Normal':
            restecg = 0
        elif restecg == 'Memiliki kelainan gelombang ST-T':
            restecg = 1
        elif restecg == 'Kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes':
            restecg = 2

        thalach = st.number_input('Masukkan Thalach (Detak Jantung Maksimum)')

        exang = st.selectbox('Exang (Angina yang Disebabkan Akibat Olahraga)', ('Tidak', 'Ya'))
        if exang == 'Tidak':
            exang = 0
        elif exang == 'Ya':
            exang = 1

        oldpeak = st.number_input('Masukkan Oldpeak')

        slope = st.selectbox('Slope (Kemiringan puncak ST)', ('Upsloping', 'Flat', 'Downsloping'))
        if slope == 'Upsloping':
            slope = 0
        elif slope == 'Flat':
            slope = 1
        elif slope == 'Downsloping':
            slope = 2
        
        ca = st.number_input('Masukkan Ca')

        thal = st.selectbox('Thal', ('Normal', 'Cacat Tetap', 'Cacat Reversibel'))
        if thal == 'Normal':
            thal = 0
        elif thal == 'Cacat Tetap':
            thal = 1
        elif thal == 'Cacat Reversibel':
            thal = 2

        model = st.selectbox(
                "Pilih Model",
                ('Gaussian Naive Bayes', 'K-Nearest Neighbor', 'Decission Tree'))


        prediksi = st.form_submit_button(label="Prediksi",type="primary")
        if prediksi:

            input = [[age,	sex,	Cp,	trestbps,	chol,	fbs,	restecg,	thalach,	exang,	oldpeak,	slope,	ca,	thal]]
            minmax = joblib.load('df_scaled.save')
            data_scaled = minmax.fit_transform(input)
            

           
            if model == 'Gaussian Naive Bayes':
                gnbclassifier = GaussianNB()
                gnbclassifier.fit(X_training, Y_training_label)
                pred = gnbclassifier.predict(input)
                st.write("Hasil Prediksi : ", model)
                if pred == 0:
                    st.write('Negatif Penyakit Jantung')
                elif pred == 1:
                    st.write('Positif Penyakit Jantung')
                    
            elif model == 'K-Nearest Neighbor':
                knnclassifier = KNeighborsClassifier(n_neighbors= 5)
                knnclassifier.fit(X_training, Y_training_label)
                pred = knnclassifier.predict(input)
                st.write("Hasil Prediksi : ", model)
                if pred == 0:
                    st.write('Negatif Penyakit Jantung')
                elif pred == 1:
                    st.write('Positif Penyakit Jantung')
                
            elif model == 'Decission Tree':
                dtclassifier = DecisionTreeClassifier()
                dtclassifier.fit(X_training, Y_training_label)
                pred = dtclassifier.predict(input)
                st.write("Hasil Prediksi Menggunakan Metode : ", model)
                if pred == 0:
                    st.write('Negatif Penyakit Jantung')
                elif pred == 1:
                    st.write('Positif Penyakit Jantung')
            


                #     st.write("Hasil Prediksi :")
                #     if pred == 0:
                #         st.write('Negatif Penyakit Jantung')
                #     elif pred == 1:
                #         st.write('Positif Penyakit Jantung')


                    # Prediksi
                    # Y_pred = knnclassifier.predict(X_test)



                    # Y_pred = gnbclassifier.predict(X_test)
                    # # hitung cm dari data testing terhadap data prediksi
                    # cm_gnb = confusion_matrix(Y_test_label, Y_pred)
                    # st.write('Hasil Confusion Matrix', cm_gnb)

                    # # Hitung Akurasi
                    # ac_gnb = round (accuracy_score(Y_test_label, Y_pred)*100)
                    # st.write("Akurasi Gaussian Naive Bayes (in %): ", ac_gnb)


            

                # modeliing
                # gnbclassifier = GaussianNB()
                # gnbclassifier.fit(X_training, Y_training_label)
                # # Y_pred = gnbclassifier.predict(X_test)
                # # # hitung cm dari data testing terhadap data prediksi
                # # cm_gnb = confusion_matrix(Y_test_label, Y_pred)
                # # st.write('Hasil Confusion Matrix', cm_gnb)

                # # # Hitung Akurasi
                # ac_gnb = round (accuracy_score(Y_test_label, Y_pred)*100)
                # # st.write("Akurasi Gaussian Naive Bayes (in %): ", ac_gnb)

                # knnclassifier = KNeighborsClassifier(n_neighbors= 5)
                # knnclassifier.fit(X_training, Y_training_label)
                # # Prediksi
                # # Y_pred = knnclassifier.predict(X_test)
                # # cm_knn = confusion_matrix(Y_test_label, Y_pred)
                # # st.write('Hasil Confusion Matrix', cm_knn)
                # ac_knn = round (accuracy_score(Y_test_label, Y_pred)*100)
                # # st.write("Akurasi KNN (in %): ", ac_knn)






