import io

import pandas as pd

CSV = b"mois,ventes,clients\njan,120,30\nfev,135,32\nmar,,35\navr,160,41\n"


def test_import_enregistre_les_metadonnees(client, upload):
    meta = upload(CSV).json()
    assert (meta["filename"], meta["rows"], meta["columns"], meta["size"]) == ("ventes.csv", 4, 3, len(CSV))
    assert client.get("/datasets/").json() == [meta]
    assert client.get(f"/datasets/{meta['id']}/").json() == meta


def test_extension_en_majuscules_acceptee(upload):
    assert upload(CSV, "VENTES.CSV").status_code == 200


def test_fichier_non_csv_refuse(upload):
    assert upload(b"x", "notes.txt").status_code == 400


def test_csv_vide_refuse_sans_laisser_de_fichier(client, upload, tmp_path):
    assert upload(b"").status_code == 400
    assert list(tmp_path.glob("*.csv")) == []
    assert client.get("/datasets/").json() == []


def test_fichier_trop_gros_refuse(upload):
    assert upload(b"a,b\n" + b"1,2\n" * 300_000).status_code == 413


def test_dataset_inconnu_renvoie_404(client):
    for url in ("/datasets/inconnu/", "/datasets/inconnu/stats/", "/datasets/inconnu/excel/",
                "/datasets/inconnu/plot/"):
        assert client.get(url).status_code == 404
    assert client.delete("/datasets/inconnu/").status_code == 404


def test_statistiques_avec_valeur_manquante(client, upload):
    stats = client.get(f"/datasets/{upload(CSV).json()['id']}/stats/").json()
    assert stats["ventes"]["count"] == 3
    assert stats["ventes"]["mean"] == (120 + 135 + 160) / 3
    assert set(stats) == {"ventes", "clients"}


def test_statistiques_nan_en_null(client, upload):
    stats = client.get(f"/datasets/{upload(b'a,b' + chr(10).encode() + b'1,x').json()['id']}/stats/").json()
    assert stats["a"]["std"] is None


def test_export_excel_relisible(client, upload):
    rep = client.get(f"/datasets/{upload(CSV).json()['id']}/excel/")
    assert rep.status_code == 200
    assert 'filename="ventes.xlsx"' in rep.headers["content-disposition"]
    assert pd.read_excel(io.BytesIO(rep.content)).shape == (4, 3)


def test_rapport_pdf(client, upload):
    rep = client.get(f"/datasets/{upload(CSV, 'données été.csv').json()['id']}/plot/")
    assert rep.status_code == 200
    assert rep.content.startswith(b"%PDF")
    assert "filename*=UTF-8''donn%C3%A9es%20%C3%A9t%C3%A9_plots.pdf" in rep.headers["content-disposition"]


def test_rapport_sans_colonne_numerique(client, upload):
    rep = client.get(f"/datasets/{upload(b'nom' + chr(10).encode() + b'a').json()['id']}/plot/")
    assert rep.status_code == 400


def test_suppression(client, upload, tmp_path):
    dataset_id = upload(CSV).json()["id"]
    assert client.delete(f"/datasets/{dataset_id}/").status_code == 200
    assert client.get("/datasets/").json() == []
    assert not (tmp_path / f"{dataset_id}.csv").exists()


def test_metadonnees_conservees_au_redemarrage(client, upload, tmp_path):
    import importlib

    import app.main
    meta = upload(CSV).json()
    importlib.reload(app.main)
    assert app.main.datasets[meta["id"]]["rows"] == 4


def test_page_d_accueil(client):
    assert client.get("/").status_code == 200
