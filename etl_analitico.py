"""Construye un CSV analítico a nivel accidente y un DW estrella SQLite.
Uso: python etl_analitico.py --entrada "zips para regis.zip" --salida salida_analitica
Dependencia: pandas. Requiere limpiar_fars.py en la misma carpeta.
"""
import argparse
import sqlite3
from pathlib import Path
import pandas as pd
from limpiar_fars import ejecutar

K = ['YEAR', 'ST_CASE']

def construir(base):
    raiz = base / 'preparacion' / 'datos_limpios' / '2024'
    def leer(t):
        return pd.read_csv(raiz / f'{t}_limpio.csv', low_memory=False)
    a = leer('accident')
    cols = K + ['STATE','STATENAME','COUNTY','COUNTYNAME','CITY','CITYNAME',
                'MONTH','DAY','DAY_WEEK','DAY_WEEKNAME','HOUR','MINUTE',
                'FECHA_ACCIDENTE','LATITUDE','LONGITUD','RUR_URB','RUR_URBNAME',
                'FUNC_SYS','FUNC_SYSNAME','WEATHER','WEATHERNAME','LGT_COND','LGT_CONDNAME',
                'RELJCT1','RELJCT1NAME','RELJCT2','RELJCT2NAME','TYP_INT','TYP_INTNAME',
                'WRK_ZONE','WRK_ZONENAME','FATALS','VE_TOTAL','VE_FORMS',
                'FECHA_INVALIDA','DIA_SEMANA_INCONSISTENTE']
    final = a[cols].copy()
    final['SINIESTROS_MORTALES'] = 1
    final['TRIMESTRE'] = ((final.MONTH - 1) // 3 + 1).astype('Int64')
    final['ES_FIN_SEMANA'] = final.DAY_WEEK.isin([1,7]).astype('Int64').mask(final.DAY_WEEK.isna())
    def franja(h):
        if pd.isna(h): return 'Desconocido'
        return ['Madrugada (00-05)', 'Mañana (06-11)', 'Tarde (12-17)', 'Noche (18-23)'][int(h)//6]
    final['FRANJA_HORARIA'] = final.HOUR.map(franja)
    final['COORDENADAS_DISPONIBLES'] = final[['LATITUDE','LONGITUD']].notna().all(axis=1).astype(int)
    final['HORA_DISPONIBLE'] = final.HOUR.notna().astype(int)
    final['MINUTO_DISPONIBLE'] = final.MINUTE.notna().astype(int)
    final['EDAD_OBSERVADA_MIN'] = pd.NA
    final = final.drop(columns='EDAD_OBSERVADA_MIN')

    def unir(res):
        nonlocal final
        final = final.merge(res, on=K, how='left', validate='one_to_one')
    v = leer('vehicle')
    # SPEEDREL: 0=no; 2/3/4/5=modalidades registradas; 8=sin conductor/conductor desconocido; 9=desconocido.
    v['VELOCIDAD_RELACIONADA'] = v.SPEEDREL.isin([2,3,4,5]).astype(int)
    v['VELOCIDAD_SIN_DATO'] = v.SPEEDREL.isna().astype(int)
    v['VELOCIDAD_NO_RELACIONADA'] = v.SPEEDREL.eq(0).astype(int)
    v['VELOCIDAD_OTRO_CODIGO'] = (v.SPEEDREL.notna() & ~v.SPEEDREL.isin([0,2,3,4,5])).astype(int)
    unir(v.groupby(K).agg(VEHICULOS_EN_CIRCULACION=('VEH_NO','size'),
        VEHICULOS_CON_VELOCIDAD_RELACIONADA=('VELOCIDAD_RELACIONADA','sum'),
        VEHICULOS_SIN_DATO_VELOCIDAD=('VELOCIDAD_SIN_DATO','sum'),
        VEHICULOS_SIN_VELOCIDAD_RELACIONADA=('VELOCIDAD_NO_RELACIONADA','sum'),
        VEHICULOS_OTRO_CODIGO_VELOCIDAD=('VELOCIDAD_OTRO_CODIGO','sum')).reset_index())
    p = leer('person')
    p['FALLECIDA'] = p.INJ_SEV.eq(4).astype(int)
    p['NO_OCUPANTE'] = p.VEH_NO.eq(0).astype(int)
    p['SEXO_SIN_DATO'] = p.SEX.isna().astype(int)
    unir(p.groupby(K).agg(PERSONAS_REGISTRADAS=('PER_NO','size'),
        FALLECIDOS_EN_PERSON=('FALLECIDA','sum'), PERSONAS_NO_OCUPANTES=('NO_OCUPANTE','sum'),
        PERSONAS_SIN_EDAD=('AGE',lambda s: s.isna().sum()),
        PERSONAS_SIN_SEXO=('SEXO_SIN_DATO','sum'),
        EDAD_MINIMA=('AGE','min'),EDAD_MAXIMA=('AGE','max'),EDAD_MEDIA=('AGE','mean')).reset_index())
    final['FALLECIDOS_CONTEO_COINCIDE'] = final.FATALS.eq(final.FALLECIDOS_EN_PERSON).astype(int)
    # Agrupar primero evita el producto cartesiano entre factores, personas y eventos.
    for t, c in {'cevent':'EVENTNUM','crashrf':'CRASHRF','distract':'DRDISTRACT',
                 'drimpair':'DRIMPAIR','driverrf':'DRIVERRF','drugs':'DRUGRES',
                 'nmdistract':'NMDISTRACT','violatn':'VIOLATION'}.items():
        d = leer(t)
        pref = t.upper()
        resumen = d.groupby(K).size().rename(pref+'_REGISTROS').reset_index()
        # Estas listas son descriptivas, no indicadores de presencia de un factor.
        etiqueta = c+'NAME'
        if etiqueta in d:
            categorias = d.groupby(K)[etiqueta].agg(
                lambda s: ' | '.join(sorted(set(s.dropna().astype(str))))).rename(pref+'_CATEGORIAS').reset_index()
            resumen = resumen.merge(categorias, on=K, validate='one_to_one')
        unir(resumen)
        final[pref+'_REGISTROS'] = final[pref+'_REGISTROS'].fillna(0).astype(int)
        if pref+'_CATEGORIAS' in final:
            final[pref+'_CATEGORIAS'] = final[pref+'_CATEGORIAS'].fillna('Sin registro en tabla auxiliar')
    enteros = ['VEHICULOS_EN_CIRCULACION','VEHICULOS_CON_VELOCIDAD_RELACIONADA',
        'VEHICULOS_SIN_DATO_VELOCIDAD','VEHICULOS_SIN_VELOCIDAD_RELACIONADA',
        'VEHICULOS_OTRO_CODIGO_VELOCIDAD','PERSONAS_REGISTRADAS','FALLECIDOS_EN_PERSON',
        'PERSONAS_NO_OCUPANTES','PERSONAS_SIN_EDAD','PERSONAS_SIN_SEXO']
    final[enteros] = final[enteros].fillna(0).astype(int)
    # VEH_NO=0 no es un vehículo; el indicador solo utiliza filas de vehicle.
    final['VELOCIDAD_RELACIONADA_ESTADO'] = 'Sin dato'
    positivo = final.VEHICULOS_CON_VELOCIDAD_RELACIONADA.gt(0)
    negativo = (final.VEHICULOS_EN_CIRCULACION.gt(0) &
                final.VEHICULOS_SIN_DATO_VELOCIDAD.eq(0) &
                final.VEHICULOS_OTRO_CODIGO_VELOCIDAD.eq(0) & ~positivo)
    final.loc[negativo,'VELOCIDAD_RELACIONADA_ESTADO'] = 'No'
    final.loc[positivo,'VELOCIDAD_RELACIONADA_ESTADO'] = 'Sí'
    assert len(final)==len(a) and not final.duplicated(K).any()
    assert final.SINIESTROS_MORTALES.sum()==len(a)
    assert final.FATALS.sum()==a.FATALS.sum()
    assert final.PERSONAS_REGISTRADAS.sum()==len(p)
    assert final.VEHICULOS_EN_CIRCULACION.sum()==len(v)
    final = final.sort_values(K)
    final.to_csv(base/'siniestros_mortales_2024_analitico.csv', index=False, encoding='utf-8-sig')
    crear_dw(final, base/'dw_seguridad_vial.sqlite')
    # Pregunta de ejemplo, sin filtrar filas con condiciones desconocidas.
    grupos = ['FRANJA_HORARIA','STATENAME','WEATHERNAME','LGT_CONDNAME','RUR_URBNAME','FUNC_SYSNAME']
    top = final.groupby(grupos,dropna=False).agg(SINIESTROS_MORTALES=('SINIESTROS_MORTALES','sum'),
                 FALLECIDOS=('FATALS','sum')).reset_index().sort_values('SINIESTROS_MORTALES',ascending=False)
    top.to_csv(base/'combinaciones_para_analisis.csv',index=False,encoding='utf-8-sig')
    dic = pd.DataFrame({'columna':final.columns,'tipo_pandas':[str(final[c].dtype) for c in final],
        'nulos':[int(final[c].isna().sum()) for c in final]})
    dic.to_csv(base/'diccionario_columnas.csv', index=False,encoding='utf-8-sig')
    print(f'CSV final: {len(final):,} accidentes, {len(final.columns)} columnas; {int(final.FATALS.sum()):,} fallecidos.')


def crear_dw(d, archivo):
    grupos = {
        'tiempo':['YEAR','MONTH','DAY','DAY_WEEK','DAY_WEEKNAME','HOUR','MINUTE','FECHA_ACCIDENTE','TRIMESTRE','ES_FIN_SEMANA','FRANJA_HORARIA'],
        'ubicacion':['STATE','STATENAME','COUNTY','COUNTYNAME','CITY','CITYNAME','LATITUDE','LONGITUD','RUR_URB','RUR_URBNAME'],
        'condicion':['WEATHER','WEATHERNAME','LGT_COND','LGT_CONDNAME'],
        'via':['FUNC_SYS','FUNC_SYSNAME','RELJCT1','RELJCT1NAME','RELJCT2','RELJCT2NAME','TYP_INT','TYP_INTNAME','WRK_ZONE','WRK_ZONENAME']}
    fact = d.copy()
    dims = {}
    for nombre, cols in grupos.items():
        sk = nombre+'_id'
        dim = d[cols].drop_duplicates().reset_index(drop=True)
        dim.insert(0,sk,range(1,len(dim)+1))
        fact = fact.merge(dim,on=cols,how='left',validate='many_to_one')
        dims[nombre] = dim
    mantener = ['YEAR','ST_CASE']+[n+'_id' for n in grupos]
    atributos = {c for cols in grupos.values() for c in cols}
    mantener += [c for c in d if c not in atributos and c not in mantener]
    fact = fact[mantener]
    with sqlite3.connect(archivo) as con:
        con.execute('PRAGMA foreign_keys=ON')
        for nombre, dim in dims.items():
            sk = nombre+'_id'
            dim.to_sql('_carga',con,index=False,if_exists='replace')
            info = con.execute('PRAGMA table_info(_carga)').fetchall()
            esquema = ','.join('"'+c[1]+'" '+('INTEGER PRIMARY KEY' if c[1]==sk else c[2]) for c in info)
            con.execute(f'CREATE TABLE dim_{nombre} ({esquema})')
            con.execute(f'INSERT INTO dim_{nombre} SELECT * FROM _carga')
            con.execute('DROP TABLE _carga')
        fact.to_sql('_carga',con,index=False,if_exists='replace')
        info = con.execute('PRAGMA table_info(_carga)').fetchall()
        esquema = ','.join('"'+c[1]+'" '+c[2]+(' NOT NULL' if c[1] in ['YEAR','ST_CASE'] or c[1].endswith('_id') else '') for c in info)
        refs = ','.join(f'FOREIGN KEY({n}_id) REFERENCES dim_{n}({n}_id)' for n in grupos)
        con.execute(f'CREATE TABLE fact_siniestro ({esquema}, PRIMARY KEY(YEAR,ST_CASE),{refs})')
        con.execute('INSERT INTO fact_siniestro SELECT * FROM _carga')
        con.execute('DROP TABLE _carga')
        for n in grupos: con.execute(f'CREATE INDEX ix_fact_{n} ON fact_siniestro({n}_id)')
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        assert con.execute('SELECT COUNT(*) FROM fact_siniestro').fetchone()[0]==len(d)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada',type=Path,required=True)
    parser.add_argument('--salida',type=Path,default=Path('salida_analitica'))
    args=parser.parse_args()
    if args.salida.exists() and any(args.salida.iterdir()):
        parser.error('La salida debe estar vacía o no existir.')
    args.salida.mkdir(parents=True,exist_ok=True)
    ejecutar(args.entrada,args.salida/'preparacion')
    construir(args.salida)
