from flask import Flask, render_template, request, redirect, flash
import mysql.connector
import pandas as pd

# MENGATASI EROR PANDAS 2.0+ 
pd.DataFrame.iteritems = pd.DataFrame.items

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json

app = Flask(__name__)
app.secret_key = "kunci_rahasia_untuk_flash_message"

# ================= KONEKSI DATABASE =================
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="db_datamining"
    )

# ================= ROUTE: BERANDA =================
@app.route('/')
def index():
    return render_template('index.html')

# ================= ROUTE: HALAMAN DATA (READ / CRUD) =================
@app.route('/data')
def lihat_data():
    page = request.args.get('page', 1, type=int)
    per_page = 100 
    offset = (page - 1) * per_page
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT COUNT(id) as total FROM transaksi")
        total_data = cursor.fetchone()['total']
        
        total_pages = (total_data + per_page - 1) // per_page
        if total_pages == 0:
            total_pages = 1
        
        # Diurutkan dari awal (ASC)
        cursor.execute("SELECT * FROM transaksi ORDER BY id ASC LIMIT %s OFFSET %s", (per_page, offset))
        data_transaksi = cursor.fetchall()
        
        cursor.close()
        conn.close()
    except Exception as e:
        data_transaksi = []
        total_pages = 1
        flash(f"Gagal mengambil data: {e}")

    return render_template('data.html', data_transaksi=data_transaksi, page=page, total_pages=total_pages)

# ================= ROUTE BARU: EDIT DATA (UPDATE) =================
@app.route('/edit', methods=['POST'])
def edit_data():
    try:
        id_data = request.form['id']
        id_transaksi = request.form['id_transaksi']
        id_customer = request.form['id_customer']
        produk = request.form['produk']
        tanggal = request.form['tanggal']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL untuk memperbarui data
        sql = """UPDATE transaksi 
                 SET id_transaksi=%s, id_customer=%s, produk=%s, tanggal=%s 
                 WHERE id=%s"""
        val = (id_transaksi, id_customer, produk, tanggal, id_data)
        
        cursor.execute(sql, val)
        conn.commit()
        cursor.close()
        conn.close()
        flash("Satu data transaksi berhasil diperbarui!")
    except Exception as e:
        flash(f"Gagal memperbarui data: {e}")
        
    # Mengarahkan kembali ke halaman data beserta nomor halamannya
    return redirect(request.referrer or '/data')

# ================= ROUTE: HAPUS DATA (DELETE) =================
@app.route('/hapus/<int:id>')
def hapus_data(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transaksi WHERE id = %s", (id,))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Satu data transaksi berhasil dihapus!")
    except Exception as e:
        flash(f"Gagal menghapus data: {e}")
    return redirect(request.referrer or '/data')

# ================= ROUTE: UPLOAD CSV (CREATE) =================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("Tidak ada file yang diunggah!")
        return redirect('/')
    
    file = request.files['file']
    if file.filename != '':
        try:
            df = pd.read_csv(file, sep=';')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE transaksi")
            
            for i, row in df.iterrows():
                sql = "INSERT INTO transaksi (id_transaksi, id_customer, produk, tanggal) VALUES (%s, %s, %s, %s)"
                val = (str(row['TransactionID']), str(row['CustomerID']), str(row['Products']), str(row['Timestamp']))
                cursor.execute(sql, val)
            
            conn.commit()
            cursor.close()
            conn.close()
            flash("Data CSV berhasil diunggah ke Database MySQL!")
        except Exception as e:
            flash(f"Terjadi kesalahan: {e}")
    return redirect('/')

# ================= ROUTE: PEMROSESAN FP-GROWTH =================
@app.route('/proses', methods=['POST'])
def proses_fpgrowth():
    tgl_mulai = request.form['tgl_mulai']
    tgl_akhir = request.form['tgl_akhir']
    
    conn = get_db_connection()
    query = f"SELECT produk FROM transaksi WHERE tanggal BETWEEN '{tgl_mulai}' AND '{tgl_akhir}'"
    df_transaksi = pd.read_sql(query, conn)
    conn.close()
    
    if df_transaksi.empty:
        flash("Tidak ada data pada rentang tanggal tersebut.")
        return redirect('/')

    # 1. Ekstraksi Rules (Data Tabel)
    transactions = [str(x).split(', ') for x in df_transaksi['produk'].tolist()]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_fp = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = fpgrowth(df_fp, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        flash("Tidak ditemukan pola dengan min support 1%.")
        return redirect('/')

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    hasil_aturan = []
    for idx, row in rules.iterrows():
        ant = ", ".join(list(row['antecedents']))
        con = ", ".join(list(row['consequents']))
        sup = round(row['support'], 3)
        conf = round(row['confidence'], 3)
        lift = round(row['lift'], 3)
        persen_conf = round(conf * 100)
        deskripsi = f"Jika pelanggan membeli {ant}, ada kemungkinan {persen_conf}% membeli {con}."
        hasil_aturan.append({'antecedents': ant, 'consequents': con, 'support': sup, 'confidence': conf, 'lift': lift, 'deskripsi': deskripsi})

    # 2. LOGIKA FP-TREE
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    min_sup_count = len(transactions) * 0.01
    frequent_items = {k: v for k, v in item_counts.items() if v >= min_sup_count}

    class FPNode:
        def __init__(self, name):
            self.name = name
            self.count = 0
            self.children = {}
            self.id = None

    root = FPNode("Root")

    for t in transactions:
        filtered = [item for item in t if item in frequent_items]
        filtered.sort(key=lambda x: frequent_items[x], reverse=True)
        
        curr = root
        for item in filtered:
            if item not in curr.children:
                curr.children[item] = FPNode(item)
            curr = curr.children[item]
            curr.count += 1

    tree_nodes = []
    tree_edges = []
    node_id_counter = 1

    def build_vis_tree(node):
        nonlocal node_id_counter
        node.id = node_id_counter
        node_id_counter += 1
        
        if len(tree_nodes) > 200: 
            return node.id
            
        label_text = f"{node.name}\n({node.count})" if node.name != "Root" else "Root"
        color = '#dc3545' if node.name == "Root" else '#17a2b8'
        
        tree_nodes.append({
            'id': node.id,
            'label': label_text,
            'shape': 'box',
            'color': {'background': color, 'border': '#117a8b'},
            'font': {'color': 'white', 'face': 'Arial'}
        })
        
        for child_name, child_node in node.children.items():
            if len(tree_nodes) <= 200:
                child_id = build_vis_tree(child_node)
                tree_edges.append({'from': node.id, 'to': child_id})
        return node.id

    build_vis_tree(root)

    return render_template('hasil.html', 
                           rules=hasil_aturan, nodes=json.dumps(tree_nodes), 
                           edges=json.dumps(tree_edges), tgl_mulai=tgl_mulai, tgl_akhir=tgl_akhir)

if __name__ == '__main__':
    app.run(debug=True)