import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/all_data.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

df_ = df.drop(['sira_no', 'kaza_il', 'tarih', 'saat', 'coord'],axis=1)
train_ = train.drop(['sira_no', 'kaza_il', 'tarih', 'saat', 'coord'],axis=1)
test_ = test.drop(['sira_no', 'kaza_il', 'tarih', 'saat', 'coord'],axis=1)

categorical_feature_mask = train_.dtypes==object
categorical_cols = train_.columns[categorical_feature_mask].tolist()

categorical_feature_mask_d =df_.dtypes==object
categorical_cols_d = df_.columns[categorical_feature_mask_d].tolist()

categorical_feature_mask_t = test_.dtypes==object
categorical_cols_t = test_.columns[categorical_feature_mask_t].tolist()

le = LabelEncoder()
train_[categorical_cols] = train_[categorical_cols].apply(lambda col: le.fit_transform(col))
train_[categorical_cols].head(10)

df_[categorical_cols_d] = df_[categorical_cols_d].apply(lambda col: le.fit_transform(col))
df_[categorical_cols_d].head(10)

test_[categorical_cols_t] =test_[categorical_cols_t].apply(lambda col: le.fit_transform(col))
test_[categorical_cols_t].head(10)
test_.rename(columns={'kaza_ilce':'enc_kaza_ilce','koy_mahalle':'enc_koy_mahalle','yol_sinifi':'enc_yol_sinifi',
                      'yol_tipi':'enc_yol_tipi','yol_kpln':'enc_yol_kpln','yerlesim_yeri':'enc_yerlesim_yeri',
                      'gun':'enc_gun','serit_banket':'enc_serit_banket','serit_durum':'enc_serit_durum',
                      'levha_durun':'enc_levha_durum','isik_durum':'enc_isik_durum','aydinlatma':'enc_aydinlatma',
                      'gun_durum':'enc_gun_durum','hava_durumu':'enc_hava_durumu','calisma_durumu':'enc_calisma_durumu',
                      'yol_cls_isaret':'enc_yol_cls_isaret','yol_yuzeyi':'enc_yol_yuzeyi','yatay_gzr':'enc_yatay_gzr',
                      'dusey_gzr':'enc_dusey_gzr','kavsak_durum':'enc_kavsak_durum','gecit_durum':'enc_gecit_durum',
                      'diger':'enc_diger','olus_1':'enc_olus_1','olus_2':'enc_olus_2','ilk_carp_yeri':'enc_ilk_carp_yeri',
                      'kaza_sonucu':'enc_kaza_sonucu','ihlal1':'enc_ihlal1','ihlal2':'enc_ihlal2','arac_cinsi':'enc_arac_cinsi',
                      'hasar_der':'enc_hasar_der','yanma':'enc_yanma','yakit':'enc_yakit',}, inplace=True)
train_.rename(columns={'kaza_ilce':'enc_kaza_ilce','koy_mahalle':'enc_koy_mahalle','yol_sinifi':'enc_yol_sinifi',
                       'yol_tipi':'enc_yol_tipi','yol_kpln':'enc_yol_kpln','yerlesim_yeri':'enc_yerlesim_yeri',
                       'gun':'enc_gun','serit_banket':'enc_serit_banket','serit_durum':'enc_serit_durum',
                       'levha_durun':'enc_levha_durum','isik_durum':'enc_isik_durum','aydinlatma':'enc_aydinlatma',
                       'gun_durum':'enc_gun_durum','hava_durumu':'enc_hava_durumu','calisma_durumu':'enc_calisma_durumu',
                       'yol_cls_isaret':'enc_yol_cls_isaret','yol_yuzeyi':'enc_yol_yuzeyi','yatay_gzr':'enc_yatay_gzr',
                       'dusey_gzr':'enc_dusey_gzr','kavsak_durum':'enc_kavsak_durum','gecit_durum':'enc_gecit_durum',
                       'diger':'enc_diger','olus_1':'enc_olus_1','olus_2':'enc_olus_2','ilk_carp_yeri':'enc_ilk_carp_yeri',
                       'kaza_sonucu':'enc_kaza_sonucu','ihlal1':'enc_ihlal1','ihlal2':'enc_ihlal2','arac_cinsi':'enc_arac_cinsi',
                       'hasar_der':'enc_hasar_der','yanma':'enc_yanma','yakit':'enc_yakit',}, inplace=True)
df_.rename(columns={'kaza_ilce':'enc_kaza_ilce','koy_mahalle':'enc_koy_mahalle','yol_sinifi':'enc_yol_sinifi',
                    'yol_tipi':'enc_yol_tipi','yol_kpln':'enc_yol_kpln','yerlesim_yeri':'enc_yerlesim_yeri',
                    'gun':'enc_gun','serit_banket':'enc_serit_banket','serit_durum':'enc_serit_durum',
                    'levha_durun':'enc_levha_durum','isik_durum':'enc_isik_durum','aydinlatma':'enc_aydinlatma',
                    'gun_durum':'enc_gun_durum','hava_durumu':'enc_hava_durumu','calisma_durumu':'enc_calisma_durumu',
                    'yol_cls_isaret':'enc_yol_cls_isaret','yol_yuzeyi':'enc_yol_yuzeyi','yatay_gzr':'enc_yatay_gzr',
                    'dusey_gzr':'enc_dusey_gzr','kavsak_durum':'enc_kavsak_durum','gecit_durum':'enc_gecit_durum',
                    'diger':'enc_diger','olus_1':'enc_olus_1','olus_2':'enc_olus_2','ilk_carp_yeri':'enc_ilk_carp_yeri',
                    'kaza_sonucu':'enc_kaza_sonucu','ihlal1':'enc_ihlal1','ihlal2':'enc_ihlal2','arac_cinsi':'enc_arac_cinsi',
                    'hasar_der':'enc_hasar_der','yanma':'enc_yanma','yakit':'enc_yakit',}, inplace=True)

df.to_csv("enc_all_data.csv", index=True)
train.to_csv("enc_train.csv", index=True)
test.to_csv("enc_test.csv", index=True)
