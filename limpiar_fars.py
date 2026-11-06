"""Limpieza conservadora y auditable de las 11 tablas FARS del ZIP 2024.
Uso: python limpiar_fars.py --entrada "zips para regis.zip" --salida resultado
También admite una carpeta con los CSV, incluida datos_originales/2024.
No descarga archivos, no modifica los originales ni une tablas de distinta granularidad.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import pandas as pd

TABLAS = ['accident', 'vehicle', 'person', 'cevent', 'crashrf', 'distract',
          'drimpair', 'driverrf', 'drugs', 'nmdistract', 'violatn']
CLAVES = {'accident': ['YEAR', 'ST_CASE'],
          'vehicle': ['YEAR', 'ST_CASE', 'VEH_NO'],
          'person': ['YEAR', 'ST_CASE', 'VEH_NO', 'PER_NO'],
          'cevent': ['YEAR', 'ST_CASE', 'EVENTNUM']}
ENTEROS = set('STATE ST_CASE VEH_NO PER_NO EVENTNUM YEAR MONTH DAY DAY_WEEK HOUR MINUTE COUNTY CITY RUR_URB FUNC_SYS WEATHER LGT_COND RELJCT1 RELJCT2 TYP_INT WRK_ZONE BODY_TYP MAKE MODEL MOD_YEAR SPEEDREL AGE SEX PER_TYP INJ_SEV SEAT_POS CRASHRF DRDISTRACT DRIMPAIR DRIVERRF DRUGSPEC DRUGRES NMDISTRACT VIOLATION'.split())
# Solo etiquetas explícitas de desconocido/no reportado. Not Applicable,
# None, Not Tested y categorías compuestas se conservan con su significado.
DESCONOCIDO = r'(?i)(?:unknown(?:\s+.*)?|not reported|reported as unknown)'


def cargar(entrada):
    if entrada.is_dir():
        archivos = [p for p in entrada.rglob('*') if p.is_file() and p.suffix.lower() == '.csv']
        fuentes = [(p.stem.lower(), p.read_bytes()) for p in archivos if p.stem.lower() in TABLAS]
    else:
        with zipfile.ZipFile(entrada) as z:
            fuentes = [(Path(n).stem.lower(), z.read(n)) for n in z.namelist()
                       if n.lower().endswith('.csv') and Path(n).stem.lower() in TABLAS]
    resultado, hashes = {}, {}
    for nombre, datos in fuentes:
        if nombre in resultado:
            raise ValueError(f'Tabla repetida en la entrada: {nombre}; seleccione un solo año.')
        for encoding in ['utf-8-sig', 'cp1252']:
            try:
                d = pd.read_csv(io.BytesIO(datos), dtype='string', encoding=encoding,
                                keep_default_na=False, on_bad_lines='error')
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f'Codificación no compatible: {nombre}')
        d.columns = d.columns.str.strip().str.upper()
        if d.columns.duplicated().any():
            raise ValueError(f'Columnas repetidas: {nombre}')
        resultado[nombre] = d
        hashes[nombre] = hashlib.sha256(datos).hexdigest()
    faltan = set(TABLAS) - resultado.keys()
    if faltan:
        raise ValueError(f'Faltan tablas: {sorted(faltan)}')
    return resultado, hashes


def ejecutar(entrada, salida):
    tablas, hashes = cargar(entrada)
    # El año se obtiene del archivo principal, nunca del nombre de la carpeta.
    years = pd.to_numeric(tablas['accident']['YEAR'], errors='raise').unique()
    if len(years) != 1 or years[0] != 2024:
        raise ValueError('Estas reglas fueron verificadas para un único año: 2024.')
    year = int(years[0])
    if entrada.resolve() == salida.resolve():
        raise ValueError('La salida debe ser distinta de la entrada.')
    # Evita mezclar resultados antiguos con una ejecución nueva.
    if salida.exists() and any(salida.iterdir()):
        raise ValueError('La carpeta de salida debe estar vacía o no existir.')
    limpio = salida / 'datos_limpios' / str(year)
    evidencia = salida / 'evidencias'
    limpio.mkdir(parents=True, exist_ok=True)
    evidencia.mkdir(parents=True, exist_ok=True)
    cambios, resumen, perfiles, relaciones = [], [], [], []

    for nombre in TABLAS:
        d = tablas[nombre]
        n = len(d)
        fila = pd.Series(range(2, n + 2), index=d.index)  # Encabezado = línea 1.

        def registrar(col, mask, regla, anterior, nuevo):
            mask = mask.fillna(False)
            for idx in d.index[mask]:
                cambios.append({'tabla': nombre, 'fila_csv': int(fila.loc[idx]),
                                'columna': col, 'regla': regla,
                                'antes': anterior.loc[idx],
                                'despues': '' if pd.isna(nuevo.loc[idx]) else nuevo.loc[idx]})

        for c in list(d.columns):
            antes = d[c].copy()
            despues = antes.str.strip().replace('', pd.NA)
            mask = antes.fillna('') != despues.fillna('')
            registrar(c, mask, 'espacios_exteriores', antes, despues)
            d[c] = despues
        vacias = d.isna().all(axis=1)
        vacias_n = int(vacias.sum())
        d = d.loc[~vacias].copy()
        # Las pruebas de drogas pueden repetirse sin identificador de prueba;
        # conservarlas evita eliminar observaciones legítimas indistinguibles.
        duplicadas = d.duplicated(keep='first') if nombre != 'drugs' else pd.Series(False, index=d.index)
        dup_n = int(duplicadas.sum())
        d.loc[duplicadas].assign(FILA_CSV=fila.loc[duplicadas]).to_csv(
            evidencia / f'{nombre}_duplicados_retirados.csv', index=False, encoding='utf-8-sig')
        d = d.loc[~duplicadas].copy()
        if 'YEAR' not in d:
            d.insert(0, 'YEAR', str(year))

        for c in list(d.columns):
            if c in ENTEROS or c in ['LATITUDE', 'LONGITUD']:
                antes = d[c].copy()
                numeric = pd.to_numeric(antes, errors='coerce')
                invalid = antes.notna() & numeric.isna()
                if c in ENTEROS:
                    invalid |= numeric.notna() & numeric.mod(1).ne(0)
                if invalid.any():
                    d[c + '_ORIGINAL'] = antes
                    numeric = numeric.mask(invalid)
                    registrar(c, invalid, 'numero_invalido', antes, numeric)
                d[c] = numeric.astype('Int64' if c in ENTEROS else 'Float64')

        # Mantiene las etiquetas y respalda el código antes de convertirlo a vacío.
        for c in list(d.columns):
            if c.endswith('NAME') or c.endswith('_ORIGINAL'):
                continue
            etiqueta = c + 'NAME'
            if etiqueta in d:
                mask = d[etiqueta].str.fullmatch(DESCONOCIDO, na=False) & d[c].notna()
                if mask.any():
                    antes = d[c].copy()
                    if c + '_ORIGINAL' not in d:
                        d[c + '_ORIGINAL'] = antes
                    d[c] = d[c].mask(mask)
                    registrar(c, mask, 'desconocido_segun_etiqueta', antes, d[c])

        rangos = {'HOUR': (0, 23), 'MINUTE': (0, 59), 'MONTH': (1, 12),
                  'DAY': (1, 31), 'DAY_WEEK': (1, 7), 'LATITUDE': (-90, 90),
                  'LONGITUD': (-180, 180)}
        for c, (lo, hi) in rangos.items():
            if c not in d:
                continue
            mask = d[c].notna() & ~d[c].between(lo, hi)
            if c == 'LATITUDE':
                # Códigos FARS detectados en el archivo; están dentro de [-90,90].
                mask |= d[c].isin([77.7777, 88.8888, 99.9999])
            if mask.any():
                antes = d[c].copy()
                if c + '_ORIGINAL' not in d:
                    d[c + '_ORIGINAL'] = antes
                d[c] = d[c].mask(mask)
                registrar(c, mask, 'rango_o_coordenada_especial', antes, d[c])

        if nombre == 'accident':
            fecha = pd.to_datetime(pd.DataFrame({'year': d.YEAR, 'month': d.MONTH,
                                                 'day': d.DAY}), errors='coerce')
            d['FECHA_ACCIDENTE'] = fecha.dt.strftime('%Y-%m-%d').astype('string')
            d['FECHA_INVALIDA'] = (d[['YEAR', 'MONTH', 'DAY']].notna().all(axis=1) & fecha.isna()).astype('Int64')
            d['DIA_SEMANA_INCONSISTENTE'] = (d.DAY_WEEK.notna() & fecha.notna() &
                d.DAY_WEEK.ne(((fecha.dt.dayofweek + 1) % 7) + 1)).fillna(False).astype('Int64')
        key = CLAVES.get(nombre)
        if key:
            problema = d[key].isna().any(axis=1) | d.duplicated(key, keep=False)
            if problema.any():
                d.loc[problema].to_csv(evidencia / f'{nombre}_conflictos_clave.csv', index=False)
                raise ValueError(f'Conflicto de clave en {nombre}. Revisar evidencia; no se eligió una fila arbitrariamente.')
        tablas[nombre] = d
        resumen.append({'tabla': nombre, 'filas_originales': n, 'filas_limpias': len(d),
                        'filas_vacias_retiradas': vacias_n, 'duplicados_exactos_retirados': dup_n,
                        'duplicados_exactos_conservados': int(d.duplicated().sum()) if nombre == 'drugs' else 0,
                        'celdas_vacias_finales': int(d.isna().sum().sum())})
        for c in d:
            perfiles.append({'tabla': nombre, 'columna': c, 'tipo_pandas': str(d[c].dtype),
                             'nulos': int(d[c].isna().sum()), 'valores_distintos': int(d[c].nunique())})
        print(f'{nombre}: {n:,} -> {len(d):,} filas', flush=True)

    def relacion(hija, padre, key, filtro=None):
        d = tablas[hija]
        elegible = pd.Series(True, index=d.index) if filtro is None else filtro
        prueba = d.loc[elegible, key].merge(tablas[padre][key].drop_duplicates(),
                                             how='left', on=key, indicator=True)
        mal = prueba['_merge'].eq('left_only').to_numpy()
        idx = d.index[elegible][mal]
        relaciones.append({'tabla': hija, 'padre': padre, 'claves': ','.join(key),
                           'filas_comprobadas': int(elegible.sum()), 'huerfanas': len(idx),
                           'filas_no_aplicables': int((~elegible).sum())})
        if len(idx):
            d.loc[idx].to_csv(evidencia / f'{hija}_sin_{padre}.csv', index=False, encoding='utf-8-sig')

    for nombre in TABLAS:
        if nombre != 'accident':
            relacion(nombre, 'accident', ['YEAR', 'ST_CASE'])
    for nombre in ['person', 'distract', 'drimpair', 'driverrf', 'violatn']:
        # VEH_NO=0 corresponde a personas no ocupantes: no exige un vehículo.
        elegible = tablas[nombre].VEH_NO.gt(0).fillna(False)
        if nombre == 'person':
            # PER_TYP=3: ocupantes de vehículos fuera de circulación, fuera de vehicle.
            elegible &= tablas[nombre].PER_TYP.ne(3).fillna(True)
        relacion(nombre, 'vehicle', ['YEAR', 'ST_CASE', 'VEH_NO'], elegible)
    for nombre in ['drugs', 'nmdistract']:
        relacion(nombre, 'person', ['YEAR', 'ST_CASE', 'VEH_NO', 'PER_NO'])
    for nombre, d in tablas.items():
        d.to_csv(limpio / f'{nombre}_limpio.csv', index=False, encoding='utf-8-sig', na_rep='')
    for archivo, datos in [('resumen_limpieza', resumen), ('perfil_columnas', perfiles),
                           ('validacion_relaciones', relaciones), ('cambios', cambios)]:
        pd.DataFrame(datos).to_csv(evidencia / f'{archivo}.csv', index=False, encoding='utf-8-sig')
    (evidencia / 'fuentes_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(f'Resultado: {salida.resolve()}')
    print('Las relaciones huérfanas se reportan y conservan; revise validacion_relaciones.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', type=Path, required=True)
    parser.add_argument('--salida', type=Path, default=Path('resultado_fars'))
    args = parser.parse_args()
    ejecutar(args.entrada, args.salida)
