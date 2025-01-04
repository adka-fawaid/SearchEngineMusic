from flask import Flask, render_template, redirect, url_for, request, session, flash
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'kunci_aplikasi_webgis_1234567890!'

# Muat data rekomendasi
musics_recomend = pickle.load(open('static/reccomend.pkl', 'rb'))
musics = pd.DataFrame(musics_recomend)
similarity = pickle.load(open('static/similarity.pkl', 'rb'))
lsa = pickle.load(open('static/lsa_model.pkl', 'rb'))
lsa_vectors = pickle.load(open('static/lsa_vectors.pkl', 'rb'))
tfidf = pickle.load(open('static/tfidf_model.pkl', 'rb'))

def fetch_track(uri):
    return f"""<iframe src="https://open.spotify.com/embed/track/{uri}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>"""

def Reccomend(music):
    music_index = musics[musics['artis_judulLagu'] == music].index[0]
    distances = similarity[music_index]
    musics_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    Reccomended_musics = []
    Reccomended_musics_tracks = []
    percentage_musics = []
    
    for i in musics_list:
        uri = musics.iloc[i[0]].uri
        # Mengubah nilai kemiripan ke persentase dan menambahkan "%" di belakangnya
        percentage_musics.append(f"{round(i[1] * 100)}%")  # Hanya persen bulat tanpa koma
        Reccomended_musics.append(musics.iloc[i[0]].artis_judulLagu)
        Reccomended_musics_tracks.append(fetch_track(uri))
        
        
    return percentage_musics, Reccomended_musics, Reccomended_musics_tracks

# Fungsi Search Engine
def SearchEngine(query, top_n=5):
    query_vector = tfidf.transform([query])
    # Transformasikan query ke ruang LSA
    query_lsa = lsa.transform(query_vector)
    
    # Hitung cosine similarity antara query dan data LSA
    query_similarity = cosine_similarity(query_lsa, lsa_vectors)
    sorted_similarities = sorted(
        list(enumerate(query_similarity[0])), 
        reverse=True, key=lambda x: x[1]
    )
    
    results = []
    # Pencarian langsung di kolom lirik
    filtered_data = musics[
    musics['Label'].str.contains(query, case=False, na=False) | 
    musics['artis_judulLagu'].str.contains(query, case=False, na=False)][['Label', 'artis_judulLagu']].head(top_n)

    for a in filtered_data.iterrows():
        index = a[0]
        song_info = {
            'artis_judulLagu': musics.iloc[index]['artis_judulLagu'],
            'track': fetch_track(musics.iloc[index]['uri']),
            'similarity' : 100  
        }
        results.append(song_info)
    for i in  sorted_similarities[:top_n]:
        index = i[0]
        similarity_score = i[1] * 100  # Mengubah ke persen
        similarity_score = round(similarity_score, 0)  # Membulatkan ke angka bulat
        song_info = {
            'artis_judulLagu': musics.iloc[index]['artis_judulLagu'],
            'track': fetch_track(musics.iloc[index]['uri']),
            'similarity': similarity_score
        }
        results.append(song_info)
    
    return results

# Fungsi untuk memverifikasi username dan password dari file CSV
def verify_user(username, password):
    users_df = pd.read_csv('admin.csv')
    user = users_df[users_df['username'] == username]
    if not user.empty and user.iloc[0]['password'] == password:
        return True
    return False

# Fungsi untuk menambah pengguna baru ke file CSV
def add_user(email, username, password, no_hp):
    users_df = pd.read_csv('admin.csv')
    new_user = pd.DataFrame({'username': [username], 'email': [email], 'password': [password], 'no_hp': [no_hp]})
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv('admin.csv', index=False)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if verify_user(username, password):  # Verifikasi user
            session['username'] = username  # Menyimpan username ke dalam session
            return redirect(url_for('home'))  # Arahkan ke home setelah login sukses

        flash('Invalid username or password')
        return render_template('login.html')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        no_hp = request.form['no_hp']

        # Tambahkan pengguna baru
        add_user(email, username, password, no_hp)
        return redirect(url_for('login'))  # Redirect ke halaman login setelah berhasil daftar

    return render_template('daftar.html')

@app.route('/logout')
def logout():
    session.clear()  # Hapus semua data sesi pengguna
    return redirect(url_for('login'))  # Arahkan kembali ke halaman login

@app.route('/home')
def home():
    if 'username' not in session:  # Cek apakah username ada dalam session
        return redirect(url_for('login'))  # Jika tidak ada, arahkan ke login
    top_songs = musics.tail(9)  # Ambil 9 lagu terbawah dari dataset

    songs_data = []

    for _, row in top_songs.iterrows():
        song_info = {
            'name': row['artis_judulLagu'],  # Nama lagu
            'track': fetch_track(row['uri'])  # Track URI dari Spotify
        }
        songs_data.append(song_info)

    return render_template('home.html', songs=songs_data)

@app.route('/search', methods=['GET', 'POST'])
def searchPage():
    if request.method == 'POST':
        query = request.form['query']  # Mendapatkan query dari form
        results = SearchEngine(query)  # Hasil pencarian menggunakan SearchEngine

        # Ambil lagu pertama dari hasil pencarian untuk rekomendasi (atau sesuaikan logika rekomendasi)
        if results:
            first_song = results[0]['artis_judulLagu']
            percentage_musics, recommended_musics, recommended_tracks = Reccomend(first_song)
        else:
            percentage_musics, recommended_musics, recommended_tracks = [], [], []

        return render_template('searchPage.html', results=results, recommended_musics=recommended_musics, 
                               recommended_tracks=recommended_tracks, percentage_musics=percentage_musics, query=query)
    return render_template('searchPage.html')

if __name__ == '__main__':
    app.run(debug=True)
