# Seguridad

## Qué nunca subir al repositorio

- `.env`, con la dirección de la cámara, su contraseña y la clave del módulo IoT.
- Claves de Roboflow o de cualquier otro servicio.
- Direcciones RTSP con usuario y contraseña reales.
- Modelos `*.pt`, videos (`grabaciones*/`), fotos (`frames*/`), `datasets/`, `runs/` y `evidence/`.

`tests/test_no_secrets.py` revisa los archivos rastreados por git y falla si encuentra una clave. Es
una ayuda y no reemplaza revisar cada cambio antes de subirlo.

## Si una clave se expone

Borrarla del archivo no basta, porque sigue en el historial de git. **Primero hay que revocarla.**

### Clave del módulo IoT (`IOT_API_KEY`)

1. Genera una nueva:

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Cámbiala en la configuración del backend.
3. Cámbiala en el `.env` del módulo.
4. Comprueba la conexión con `python -m safevision_iot check-backend`.

### Clave de Roboflow

1. En Roboflow, abre **Settings → API Keys**.
2. Revoca la clave expuesta y genera otra.
3. Actualiza el secreto `ROBOFLOW_API_KEY` en Google Colab.

### Contraseña de la cámara

Cámbiala en la cuenta de cámara de la app Tapo y actualiza `CAMERA_RTSP_URL` en `.env`.

## Secretos de Google Colab

Los cuadernos leen la clave con `userdata.get("ROBOFLOW_API_KEY")`:

1. Abre el panel de secretos con el ícono de llave.
2. Pulsa **Add new secret**.
3. Escribe el nombre `ROBOFLOW_API_KEY` y pega la clave como valor.
4. Activa el acceso para el cuaderno.

**Nunca pegues tu clave en el cuaderno ni lo guardes en GitHub con ella.**

## Video en vivo y `VIDEO_HOST`

| Valor | Quién puede conectarse |
|---|---|
| `127.0.0.1` (por defecto) | Solo este PC |
| `0.0.0.0` | Cualquier equipo de la red local; úsalo solo en una red de confianza |

El servidor **no tiene contraseña** y permite solicitudes desde cualquier origen. Por eso, aun con
`127.0.0.1`, una página web abierta en este PC podría pedir la imagen si el navegador le da permiso
de acceso a la red local. Concede ese permiso solo al panel de SafeVision AI.

## Privacidad

- El video se analiza en el computador local y no se envía al backend.
- Las fotos de evidencia pueden mostrar personas y menores de edad. Se guardan en `evidence/`, fuera
  del repositorio; limita quién accede a esa carpeta y borra las que ya no necesites.
- Los videos usados para entrenar y evaluar no se publican.
