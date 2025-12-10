# JUSTIFICACIÓN DE LAS 41 CARACTERÍSTICAS (FEATURES) DEL IDS

## 📋 Introducción

El modelo IDS utiliza **41 características de red** del dataset NSL-KDD para clasificar si una conexión es:
- **Normal (0)**: Tráfico legítimo
- **Ataque (1)**: Intrusión o actividad maliciosa

Cada característica fue seleccionada porque:
1. **Captura anomalías**: Detecta desviaciones del comportamiento normal
2. **Diferencia ataques**: Tiene valores distintos en ataques vs tráfico normal
3. **Es computacionalmente eficiente**: Fácil de extraer en tiempo real

---

## 🔍 DESGLOSE DE TODAS LAS 41 CARACTERÍSTICAS

### **GRUPO 1: CARACTERÍSTICAS BÁSICAS DE CONEXIÓN (4 features)**

#### 1. `duration`
```
Valor: Número (segundos)
Rango: 0 a 58,329 segundos

¿QUÉ MIDE?
  └─ Duración total de la conexión en segundos

¿POR QUÉ ES IMPORTANTE?
  • DoS attacks: conexiones MUY CORTAS (< 1 segundo)
    Atacante envía miles de paquetes SYN y cierra rápido
  
  • R2L attacks: conexiones LARGAS (intentos de login)
    Atacante intenta múltiples contraseñas
  
  • Normal traffic: duración VARIABLE (depende del servicio)
    HTTP: segundos, FTP: minutos, SSH: horas

PATRÓN:
  DoS:     duration ↓↓ (muy corto)
  Normal:  duration ↔ (variable, servicio-dependiente)
  R2L:     duration ↑ (largo por intentos)
```

#### 2. `protocol_type`
```
Valor: TCP, UDP, ICMP (categoría → convertida a número)
Ejemplo: TCP=0, UDP=1, ICMP=2

¿QUÉ MIDE?
  └─ Protocolo de capa de transporte usado

¿POR QUÉ ES IMPORTANTE?
  • TCP: Protocolo confiable (conexiones establecidas)
    - Usado por: SSH, HTTP, FTP, SMTP
    - Ataques típicos: R2L, U2R (requieren sesión confiable)
  
  • UDP: Protocolo sin conexión (datagramas)
    - Usado por: DNS, NTP, DHCP
    - Ataques típicos: DoS (UDP Flood es rápido)
  
  • ICMP: Protocolo de control (ping, traceroute)
    - Usado por: Diagnóstico de red
    - Ataques típicos: Ping Flood, ICMP redirect

PATRÓN:
  DoS UDP:   protocol_type = UDP (muchos datagramas)
  Probe:     protocol_type = ICMP (ping scan)
  R2L/U2R:   protocol_type = TCP (sesión confiable)
  Normal:    MIXTO (depende del servicio)
```

#### 3. `service`
```
Valor: http, ftp, ssh, telnet, smtp, pop3, dns, etc.
(categoría → convertida a número con LabelEncoder)

¿QUÉ MIDE?
  └─ Puerto/servicio destino de la conexión

¿POR QUÉ ES IMPORTANTE?
  • Servicios expuestos = objetivos de ataque
  
  • Servicios de autenticación = objetivo R2L
    - ssh (22): Fuerza bruta SSH
    - ftp (21): Fuerza bruta FTP
    - telnet (23): Acceso remoto sin encriptación
  
  • Servicios web = objetivo Probe + DoS
    - http (80): Web servers
    - https (443): Secure web
  
  • Servicios de correo = objetivo R2L
    - smtp (25): Envío de correos
    - pop3 (110): Descarga de correos

PATRÓN:
  DoS:      service = http, https (sitios populares)
  Probe:    service = VARIABLE (escanea múltiples servicios)
  R2L:      service = ssh, ftp, telnet (autenticación)
  U2R:      service = shell, exec (comando remoto)
  Normal:   service = ESPERADO (usuario accede su servicio habitual)
```

#### 4. `flag`
```
Valor: S0, S1, S2, S3, SF, REJ, RSTO, RSTR, RSTOS0, OTH
(estados de conexión TCP)

¿QUÉ MIDE?
  └─ Estado de la conexión (flags TCP)

¿QÚAL ES LA TABLA DE FLAGS?

  S0  = Conexión rechazada (no SYN-ACK desde servidor)
  S1  = SYN enviado (cliente esperando respuesta)
  S2  = SYN recibido del servidor
  S3  = SYN enviado/recibido (establecida)
  SF  = Session Finished (conexión completada normalmente)
  REJ = Conexión rechazada por servidor
  RSTO = Reset de servidor
  RSTR = Reset de cliente
  RSTOS0 = Reset del servidor sin ACK previo
  OTH = Otros (flags no clasificados)

¿POR QUÉ ES IMPORTANTE?
  • Flag S0: Indicador de DoS (muchas conexiones incompletas)
    └─ Atacante envía SYN, servidor responde SYN-ACK, 
       pero atacante NO responde ACK → conexión "colgada"
  
  • Flag SF: Conexión normal (se completó)
    └─ Cliente y servidor cerraron conexión ordenadamente
  
  • Flag REJ: Servidor rechazó conexión
    └─ Indicador de Probe (escaneo de puertos cerrados)
  
  • Flag RSTO: Reset del servidor
    └─ Servidor cerró conexión bruscamente (sospechoso)

PATRÓN:
  DoS:     flag = S0 (muchas conexiones incompletas)
  Probe:   flag = REJ (muchos puertos rechazados)
  Normal:  flag = SF (sesiones completadas normalmente)
```

---

### **GRUPO 2: VOLUMEN DE TRÁFICO (2 features)**

#### 5. `src_bytes`
```
Valor: Bytes enviados por el origen (cliente/atacante)
Rango: 0 a 4,294,967,295 bytes

¿QUÉ MIDE?
  └─ Cantidad de datos enviados POR EL ORIGEN

¿POR QUÉ ES IMPORTANTE?
  • DoS attacks: src_bytes ↑↑↑ (mucho tráfico)
    Atacante envía gigabytes de datos para saturar servidor
    Ejemplo: 1GB en 10 segundos = ataque claro
  
  • Probe attacks: src_bytes ↓ (poco tráfico)
    Atacante solo envía paquetes pequeños de prueba
  
  • R2L attacks: src_bytes VARIABLE
    Intenta login (pocas bytes inicialmente)
  
  • Normal: src_bytes VARIABLE (depende de servicio)
    HTTP POST: muchos bytes (subida de archivos)
    HTTP GET: pocos bytes (solo request)

PATRÓN:
  DoS:     src_bytes >> 1MB (mucho tráfico del atacante)
  Probe:   src_bytes << 1KB (reconocimiento, poco tráfico)
  Normal:  src_bytes SERVICIO-DEPENDIENTE
```

#### 6. `dst_bytes`
```
Valor: Bytes enviados por el destino (servidor)
Rango: 0 a 4,294,967,295 bytes

¿QUÉ MIDE?
  └─ Cantidad de datos enviados POR EL SERVIDOR

¿POR QUÉ ES IMPORTANTE?
  • DoS attacks: dst_bytes ↓ (poco, servidor abrumado)
    Servidor no puede responder a todos los paquetes
  
  • Normal transfers: dst_bytes ↑ (servidor responde)
    Descarga de archivo: dst_bytes ↑↑↑
    Consulta DB: dst_bytes ↑
  
  • Probe attacks: dst_bytes ↓ (poco tráfico)
    Servidor rechaza conexiones rápidamente
  
  • R2L attacks: dst_bytes VARIABLE
    Servidor responde con prompts de login, etc.

PATRÓN:
  DoS:     dst_bytes ↓ (servidor no puede responder)
  Probe:   dst_bytes ↓ (respuestas de rechazo pequeñas)
  Normal:  dst_bytes ↑ (servidor responde activamente)
```

---

### **GRUPO 3: ANOMALÍAS DETECTADAS (5 features)**

#### 7. `land`
```
Valor: 0 (normal) o 1 (sospechoso)
Definición: ¿El origen (source) y destino (destination) son la MISMA IP?

¿QUÉ MIDE?
  └─ Si la conexión es de una IP hacia sí misma

¿POR QUÉ ES IMPORTANTE?
  • EXTREMADAMENTE SOSPECHOSO en redes reales
  • Ataque conocido: "Land Attack"
    └─ Enviar paquetes SYN con origen = destino
    └─ Servidor entra en loop infinito
    └─ Resultado: DoS o crash del servidor
  
  • Casos normales: CASI NUNCA (quizás localhost testing)
  
  • Si land = 1: BANDERA ROJA 🚩
    └─ Probabilidad muy alta de ataque

PATRÓN:
  DoS (Land Attack):  land = 1 (100% indicador)
  Normal:             land = 0 (siempre)
```

#### 8. `wrong_fragment`
```
Valor: Número de fragmentos de IP incorrectos
Rango: 0 a 3

¿QUÉ MIDE?
  └─ Número de fragmentos IP malformados en la conexión

¿QUÉ SON FRAGMENTOS IP?
  • IP puede fragmentar paquetes grandes en múltiples fragmentos
  • Campo "fragment offset" indica posición del fragmento
  • "Wrong fragment" = offset indicando superposición

¿POR QUÉ ES IMPORTANTE?
  • ANOMALÍA: fragmentos incorrectos NO deben ocurrir en tráfico normal
  • Indicador de: Ataque de fragmentación (evasión IDS)
  • Técnica de evasión: Enviar fragmentos malformados para confundir IDS/Firewall
  
  • Si wrong_fragment > 0: SOSPECHOSO 🚩

PATRÓN:
  Probe/Evasión:  wrong_fragment > 0 (anomalía técnica)
  Normal:         wrong_fragment = 0
```

#### 9. `urgent`
```
Valor: Número de paquetes con bit "urgent" activado
Rango: 0 a 14

¿QUÉ MIDE?
  └─ Número de paquetes con URG flag (urgent data)

¿QUÉ ES EL URGENT FLAG?
  • Flag TCP que indica: "Los siguientes datos son urgentes"
  • Usado por: Aplicaciones antiguas (telnet, rsh)
  • En redes modernas: MUY RARO

¿POR QUÉ ES IMPORTANTE?
  • ANOMALÍA: uso excesivo de urgent = sospechoso
  • Puede indicar: Ataque de fragmentación o evasión
  • Tráfico normal moderno: urgent ≈ 0 siempre
  
  • Si urgent > 0: SOSPECHOSO 🚩

PATRÓN:
  Evasión/Probe:  urgent > 0 (técnica antigua/evasión)
  Normal:         urgent = 0
```

#### 10. `hot`
```
Valor: Número de conexiones a puertos "hot" (sensibles)
Rango: 0 a 255

¿QUÉ MIDE?
  └─ Intentos de acceso a puertos "calientes" (sensibles)

¿CUÁLES SON LOS PUERTOS "HOT"?
  • Puertos privilegiados / servicios administrativos
  • Ejemplos: telnet (23), SMTP (25), exec (512), login (513), shell (514)
  • Puertos donde intentar acceso sin autorización = ATAQUE

¿POR QUÉ ES IMPORTANTE?
  • Indicador de R2L / U2R (escalada de privilegios)
  • Atacante busca acceder a servicios administrativos
  • Si hot > 0: Probablemente intento de acceso no autorizado 🚩
  
  • Usuarios normales: no acceden a estos puertos

PATRÓN:
  R2L/U2R:  hot > 0 (intento de acceso privilegiado)
  Normal:   hot = 0
```

#### 11. `num_failed_logins`
```
Valor: Número de intentos fallidos de login
Rango: 0 a 5

¿QUÉ MIDE?
  └─ Cuántas veces falló el login en esta conexión

¿POR QUÉ ES IMPORTANTE?
  • Indicador DIRECTO de R2L (fuerza bruta)
  • Atacante prueba múltiples contraseñas
  
  • Patrón de fuerza bruta:
    └─ num_failed_logins = 0, 1, 2, 3, 4, luego num_failed_logins = 5 (éxito)
  
  • Si num_failed_logins > 0: PROBABILIDAD ALTA de ataque 🚩
  
  • Usuarios normales: casi nunca fallan en login
    (máximo 1-2 veces si olvidan contraseña)

PATRÓN:
  R2L (Fuerza bruta):  num_failed_logins = 1, 2, 3, 4, 5
  Normal:              num_failed_logins = 0
```

---

### **GRUPO 4: CARACTERÍSTICAS DE SESIÓN (3 features)**

#### 12. `logged_in`
```
Valor: 0 (no) o 1 (sí)

¿QUÉ MIDE?
  └─ ¿La conexión logró login exitosamente?

¿POR QUÉ ES IMPORTANTE?
  • R2L attacks: logged_in = 1 DESPUÉS de num_failed_logins > 0
    └─ Patrón: Fuerza bruta → éxito
  
  • Normal users: logged_in = 1 (login normal)
  
  • Probe/DoS: logged_in = 0 (nunca autenticaron)

PATRÓN:
  R2L:   num_failed_logins > 0 AND logged_in = 1 (éxito tras intentos)
  DoS:   logged_in = 0 (nunca entró)
  Normal: logged_in = 1 (entrada normal)
```

#### 13. `num_compromised`
```
Valor: Número de hosts comprometidos detectados
Rango: 0 a 7

¿QUÉ MIDE?
  └─ Cuántos hosts comprometidos fue accedido en la conexión

¿POR QUÉ ES IMPORTANTE?
  • U2R attacks: num_compromised > 0 (atacante accedió a hosts)
  • Indicador de: Movimiento lateral en la red
  
  • Si num_compromised > 0: ATAQUE GRAVE 🚩
    └─ Red ya comprometida, atacante moviéndose

PATRÓN:
  U2R:    num_compromised > 0 (escalada y movimiento)
  Normal: num_compromised = 0
```

#### 14. `root_shell`
```
Valor: 0 (no) o 1 (sí)

¿QUÉ MIDE?
  └─ ¿Se obtuvo acceso root/administrador?

¿POR QUÉ ES IMPORTANTE?
  • INDICADOR CRÍTICO de U2R (escalada de privilegios)
  • Si root_shell = 1: ATAQUE EXITOSO 🚩🚩🚩
    └─ Atacante tiene control total del sistema
  
  • Usuarios normales: root_shell = 0
    (un usuario normal NO accede como root)

PATRÓN:
  U2R (exitoso):  root_shell = 1
  Normal:         root_shell = 0
```

---

### **GRUPO 5: CREACIÓN DE ARCHIVOS / PERMISOS (3 features)**

#### 15. `su_attempted`
```
Valor: 0 (no) o 1 (sí)

¿QUÉ MIDE?
  └─ ¿Se intentó comando "su" (switch user a root)?

¿QUÉ ES "su"?
  • Comando Unix/Linux: "su" = "switch user"
  • "su root" = cambiar a usuario root
  • Requiere contraseña de root

¿POR QUÉ ES IMPORTANTE?
  • INDICADOR de U2R (escalada de privilegios)
  • Si su_attempted = 1: Probable intento de escalada 🚩
  
  • Usuarios normales: su_attempted = 0
    (usuarios normales no necesitan cambiar a root)

PATRÓN:
  U2R:    su_attempted = 1 (intento de escalada)
  Normal: su_attempted = 0
```

#### 16. `num_shells`
```
Valor: Número de shells abiertos
Rango: 0 a 5

¿QUÉ MIDE?
  └─ Cuántos shells se abrieron en la sesión

¿QUÉ ES UN SHELL?
  • Shell = línea de comandos (bash, sh, csh, etc.)
  • Atacante abre shell para ejecutar comandos

¿POR QUÉ ES IMPORTANTE?
  • Indicador de U2R (ejecutar comandos como root)
  • Si num_shells > 0: ACTIVIDAD EJECUTIVA 🚩
    └─ Alguien ejecutó múltiples comandos
  
  • Conexiones normales (HTTP, FTP): num_shells = 0
    (no abren shells interactivas)

PATRÓN:
  U2R:    num_shells > 0 (ejecución de comandos)
  Normal: num_shells = 0
```

#### 17. `num_access_files`
```
Valor: Número de archivos accedidos
Rango: 0 a 8

¿QUÉ MIDE?
  └─ Cuántos archivos fueron accedidos/modificados

¿POR QUÉ ES IMPORTANTE?
  • Indicador de R2L/U2R (exploración de archivos)
  • Si num_access_files > 0: EXPLORACIÓN 🚩
    └─ Atacante buscando archivos interesantes (credenciales, datos)
  
  • Conexiones normales (HTTP): num_access_files = 0 típicamente
    (web servers no tienen "acceso a archivos" en este sentido)

PATRÓN:
  R2L/U2R:  num_access_files > 0 (recopilación de datos)
  Normal:   num_access_files = 0
```

#### 18. `num_outbound_cmds`
```
Valor: Número de comandos salientes ejecutados
Rango: 0 a 0 (siempre 0 en NSL-KDD)

¿QUÉ MIDE?
  └─ Cuántos comandos se enviaron desde el servidor

NOTA: En el dataset NSL-KDD, esta columna SIEMPRE es 0
(No hay datos de comandos salientes)

¿POR QUÉ ESTÁ?
  • Incluida por completitud del dataset KDD original
  • Sería importante si hubiera datos (detectar reverse shell)

PATRÓN:
  (No aplicable en NSL-KDD)
```

---

### **GRUPO 6: ESTADÍSTICAS DE CONEXIÓN LOCALES (8 features)**

Estas 8 features miran la **conexión actual** dentro de una **ventana de 2 segundos**.

#### 19. `count`
```
Valor: Número de conexiones al mismo host destino
Rango: 1 a 511

¿QUÉ MIDE?
  └─ En los últimos 2 segundos, ¿cuántas conexiones 
     al MISMO HOST DESTINO desde el MISMO ORIGEN?

¿POR QUÉ ES IMPORTANTE?
  • DoS attacks: count ↑↑↑ (muchas conexiones rápidas)
    Atacante envía 100s-1000s de paquetes/segundo
  
  • Probe attacks: count ↑ (múltiples intentos de conexión)
    Escaneo de puertos: intenta 65,535 puertos
  
  • Normal users: count = 1-10 (conexiones esporádicas)

PATRÓN:
  DoS:     count > 100 (saturation)
  Probe:   count = 10-100 (scanning)
  Normal:  count = 1-5 (typical)
```

#### 20. `srv_count`
```
Valor: Número de conexiones al MISMO SERVICIO
Rango: 1 a 511

¿QUÉ MIDE?
  └─ En los últimos 2 segundos, ¿cuántas conexiones 
     al MISMO PUERTO/SERVICIO desde CUALQUIER origen?

¿DIFERENCIA CON `count`?
  • count: mismo host destino
  • srv_count: mismo servicio (cualquier host)

¿POR QUÉ ES IMPORTANTE?
  • DoS attacks (multi-source): srv_count ↑↑↑
    Múltiples atacantes → mismo servicio
  
  • Probe attacks: srv_count ↑ (múltiples puertos)
  
  • Normal: srv_count = 1-20 (usuarios normales usan mismo servicio)

PATRÓN:
  DoS:     srv_count > 100
  Normal:  srv_count = 1-50
```

#### 21. `serror_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones con ERROR SYN en `count`

¿QUÉ MIDE?
  └─ De las últimas N conexiones (count),
     ¿cuántas tuvieron error SYN?

¿QUÉ ES ERROR SYN?
  • Conexión TCP que NO completó handshake SYN-ACK
  • Servidor responde SYN-ACK, pero cliente no responde ACK
  • Conexión "colgada"

¿POR QUÉ ES IMPORTANTE?
  • DoS attacks: serror_rate ↑ (muchas conexiones incompletas)
    Técnica: SYN Flood
  
  • Normal: serror_rate ≈ 0% (conexiones siempre se completan)

PATRÓN:
  SYN Flood:  serror_rate > 0.5 (> 50% errores)
  Normal:     serror_rate ≈ 0.0
```

#### 22. `srv_serror_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones con ERROR SYN en `srv_count`

¿DIFERENCIA CON `serror_rate`?
  • serror_rate: dentro de mismo host destino
  • srv_serror_rate: dentro de mismo servicio

¿POR QUÉ ES IMPORTANTE?
  • Similar a serror_rate
  • Útil para detectar DoS multi-host contra mismo servicio

PATRÓN:
  SYN Flood (multi-host):  srv_serror_rate > 0.5
  Normal:                  srv_serror_rate ≈ 0.0
```

#### 23. `rerror_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones RECHAZADAS en `count`

¿QUÉ MIDE?
  └─ De las últimas N conexiones (count),
     ¿cuántas fueron RECHAZADAS (REJ flag)?

¿POR QUÉ ES IMPORTANTE?
  • Probe attacks: rerror_rate ↑ (escaneo de puertos cerrados)
    Atacante intenta puertos, la mayoría rechazados
  
  • DoS attacks: rerror_rate ↓ (intenta abrumar)
    No le importa si rechazadas, solo saturar
  
  • Normal: rerror_rate ≈ 0% (conexiones aceptadas)

PATRÓN:
  Probe:   rerror_rate > 0.5 (muchos puertos cerrados)
  DoS:     rerror_rate ≈ 0.0
  Normal:  rerror_rate ≈ 0.0
```

#### 24. `srv_rerror_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones RECHAZADAS en `srv_count`

Similar a `rerror_rate` pero para `srv_count`.
```

#### 25. `same_srv_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones al MISMO SERVICIO en `count`

¿QUÉ MIDE?
  └─ De las últimas N conexiones (count),
     ¿cuántas fueron al MISMO PUERTO/SERVICIO?

¿POR QUÉ ES IMPORTANTE?
  • Normal users: same_srv_rate ↑ (acceden mismo servicio)
    Usuario HTTP: siempre puerto 80
  
  • Probe attacks: same_srv_rate ↓ (múltiples puertos)
    Escaneo de puertos: intenta todos
  
  • DoS: same_srv_rate ↑ o ↓ (depende del objetivo)

PATRÓN:
  Probe:   same_srv_rate < 0.5 (muchos puertos diferentes)
  Normal:  same_srv_rate ↑ (mismo puerto típico)
```

#### 26. `diff_srv_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones a SERVICIOS DIFERENTES en `count`

¿DIFERENCIA CON `same_srv_rate`?
  • same_srv_rate: % mismo servicio
  • diff_srv_rate: % DIFERENTES servicios
  • Suma: same_srv_rate + diff_srv_rate ≈ 1.0

¿POR QUÉ ES IMPORTANTE?
  • Probe attacks: diff_srv_rate ↑ (escaneo de puertos)
  • Normal: diff_srv_rate ↓ (mismo servicio típico)

PATRÓN:
  Probe:   diff_srv_rate > 0.5 (muchos puertos)
  Normal:  diff_srv_rate < 0.2 (poca variedad)
```

#### 27. `srv_diff_host_rate`
```
Valor: Tasa/Porcentaje (0.0 a 1.0)
Definición: % de conexiones a HOSTS DIFERENTES en `srv_count`

¿QUÉ MIDE?
  └─ De las últimas N conexiones al MISMO SERVICIO,
     ¿cuántas fueron a HOSTS DIFERENTES?

¿POR QUÉ ES IMPORTANTE?
  • Probe attacks: srv_diff_host_rate ↑ (múltiples objetivos)
    Network mapping: escanea múltiples IPs
  
  • Normal: srv_diff_host_rate ↓ (mismo servidor típico)
    Usuario conecta al server HTTP de la empresa

PATRÓN:
  Probe:   srv_diff_host_rate > 0.5
  Normal:  srv_diff_host_rate < 0.2
```

---

### **GRUPO 7: ESTADÍSTICAS DE HOST DESTINO (9 features)**

Estas 9 features miran a TODOS los hosts destino en las **últimas 100 conexiones**.

#### 28. `dst_host_count`
```
Valor: Número de conexiones al host destino
Rango: 1 a 255

¿QUÉ MIDE?
  └─ En las últimas 100 conexiones,
     ¿cuántas fueron HACIA ESTE HOST DESTINO?

¿POR QUÉ ES IMPORTANTE?
  • DoS targets: dst_host_count ↑↑↑
    Host popular siendo atacado
  
  • Normal hosts: dst_host_count VARIABLE
    Servidores populares: count ↑
    Servidores internos: count ↓

PATRÓN:
  DoS target:  dst_host_count > 200
  Normal:      dst_host_count = 1-100
```

#### 29. `dst_host_srv_count`
```
Valor: Número de conexiones al MISMO SERVICIO del host destino
Rango: 1 a 255

¿DIFERENCIA CON `dst_host_count`?
  • dst_host_count: todas las conexiones al host
  • dst_host_srv_count: conexiones al MISMO PUERTO del host

¿POR QUÉ ES IMPORTANTE?
  • Detecta si atacante enfoca PUERTO ESPECÍFICO del host
  
  • DoS HTTP: dst_host_srv_count ↑↑↑ (puerto 80 siendo atacado)

PATRÓN:
  DoS (puerto específico):  dst_host_srv_count > 200
```

#### 30. `dst_host_same_srv_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones al MISMO SERVICIO en `dst_host_count`

¿QUÉ MIDE?
  └─ De las últimas 100 conexiones AL HOST,
     ¿cuántas fueron al MISMO PUERTO?

¿POR QUÉ ES IMPORTANTE?
  • Normal services: same_srv_rate ↑ (mismo puerto típico)
    HTTP service: 99% port 80
  
  • Probe/Scan: same_srv_rate ↓ (múltiples puertos)
    Port scan: 1% port 80, 1% port 22, ..., etc.

PATRÓN:
  Probe:   dst_host_same_srv_rate < 0.3
  Normal:  dst_host_same_srv_rate > 0.7
```

#### 31. `dst_host_diff_srv_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones a SERVICIOS DIFERENTES en `dst_host_count`

Opuesto a `dst_host_same_srv_rate`.

PATRÓN:
  Probe:   dst_host_diff_srv_rate > 0.7
  Normal:  dst_host_diff_srv_rate < 0.3
```

#### 32. `dst_host_same_src_port_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones desde MISMO PUERTO ORIGEN

¿QUÉ MIDE?
  └─ De las últimas 100 conexiones AL HOST,
     ¿cuántas vinieron del MISMO PUERTO ORIGEN?

¿POR QUÉ ES IMPORTANTE?
  • Normal: same_src_port_rate ↑
    Usuario abre sesión desde puerto efímero X, reutiliza
  
  • Probe/Random: same_src_port_rate ↓
    Atacante usa puertos aleatorios para cada intento

PATRÓN:
  Normal:  dst_host_same_src_port_rate > 0.5
  Probe:   dst_host_same_src_port_rate < 0.3
```

#### 33. `dst_host_srv_diff_host_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones desde HOSTS DIFERENTES (a MISMO SERVICIO)

¿QUÉ MIDE?
  └─ De las últimas 100 conexiones al MISMO PUERTO del HOST,
     ¿cuántas vinieron de HOSTS DIFERENTES?

¿POR QUÉ ES IMPORTANTE?
  • Network scan: srv_diff_host_rate ↑ (múltiples orígenes)
    Distributed scan o botnet
  
  • Normal: srv_diff_host_rate ↓ (mismo origen típico)
    Usuarios internos → servidor centralizado

PATRÓN:
  Distributed attack:  dst_host_srv_diff_host_rate > 0.5
  Normal:              dst_host_srv_diff_host_rate < 0.3
```

#### 34. `dst_host_serror_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones con ERROR SYN en `dst_host_count`

¿QUÉ MIDE?
  └─ De las últimas 100 conexiones AL HOST,
     ¿cuántas tuvieron error SYN?

¿POR QUÉ ES IMPORTANTE?
  • SYN Flood atacando host: serror_rate ↑↑↑ (> 0.5)
  • Normal: serror_rate ≈ 0

PATRÓN:
  SYN Flood (target):  dst_host_serror_rate > 0.5
  Normal:              dst_host_serror_rate ≈ 0.0
```

#### 35. `dst_host_srv_serror_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones con ERROR SYN en `dst_host_srv_count`

Similar a `dst_host_serror_rate` pero para MISMO PUERTO del host.

PATRÓN:
  SYN Flood (port):  dst_host_srv_serror_rate > 0.5
```

#### 36. `dst_host_rerror_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones RECHAZADAS en `dst_host_count`

¿POR QUÉ ES IMPORTANTE?
  • Probe/Scan: rerror_rate ↑ (muchos puertos cerrados)
  • Normal: rerror_rate ≈ 0 (conexiones aceptadas)

PATRÓN:
  Probe:   dst_host_rerror_rate > 0.5
  Normal:  dst_host_rerror_rate ≈ 0.0
```

#### 37. `dst_host_srv_rerror_rate`
```
Valor: Tasa (0.0 a 1.0)
Definición: % de conexiones RECHAZADAS en `dst_host_srv_count`

Similar a `dst_host_rerror_rate` pero para MISMO PUERTO.
```

---

### **GRUPO 8: VARIABLES OBJETIVO (4 features - NO USADAS EN ENTRENAMIENTO)**

#### 38. `protocol_type` (REPETIDA - IDENTIFICADOR)
#### 39. `service` (REPETIDA - IDENTIFICADOR)
#### 40. `label` ✅ **ETIQUETA OBJETIVO**
```
Valor: 'normal' o 'attack' (convertida a 0/1)

¿QUÉ MIDE?
  └─ Clasificación correcta de la conexión

¿POR QUÉ EXISTE?
  • Es la variable que el modelo APRENDE A PREDECIR
  • El modelo recibe X (41 features) → predice Y (label)

NOTA: En producción, NO TENEMOS esta etiqueta
(el IDS debe PREDECIRLA basándose en 41 features)
```

#### 41. `difficulty`
```
Valor: Número (dificultad de clasificación)
Rango: 1-21 (no usada en NSL-KDD mejorado)

¿QUÉ MIDE?
  └─ Dificultad de clasificar correctamente la muestra

¿POR QUÉ NO LA USAMOS?
  • Información que no estaría disponible en producción
  • Se descarta al preparar datos (drop en el código)

PATRÓN:
  (No aplicable - descartada)
```

---

## 📊 TABLA RESUMIDA: JUSTIFICACIÓN POR TIPO DE ATAQUE

```
┌─────────────┬──────────────────────────────────────────────────────────┐
│ TIPO ATAQUE │ FEATURES CLAVE                                           │
├─────────────┼──────────────────────────────────────────────────────────┤
│             │                                                          │
│ DoS         │ • duration ↓ (conexiones cortas)                         │
│             │ • count ↑↑↑ (muchas conexiones rápidas)                  │
│             │ • src_bytes ↑↑↑ (mucho tráfico del atacante)             │
│             │ • dst_bytes ↓ (servidor no puede responder)              │
│             │ • serror_rate ↑ (errores SYN - SYN Flood)               │
│             │ • flag = S0 (conexiones incompletas)                     │
│             │                                                          │
├─────────────┼──────────────────────────────────────────────────────────┤
│             │                                                          │
│ Probe       │ • diff_srv_rate ↑ (múltiples puertos)                   │
│             │ • rerror_rate ↑ (puertos rechazados)                     │
│             │ • dst_bytes ↓ (poco tráfico efectivo)                    │
│             │ • flag = REJ (rechazos)                                  │
│             │ • wrong_fragment > 0 (fragmentación anómala)             │
│             │                                                          │
├─────────────┼──────────────────────────────────────────────────────────┤
│             │                                                          │
│ R2L         │ • num_failed_logins > 0 (intentos fallidos)             │
│             │ • logged_in = 1 DESPUÉS (éxito tras fallos)             │
│             │ • service = ssh, ftp, telnet (autenticación)            │
│             │ • duration ↑ (conexión larga)                           │
│             │ • hot > 0 (puertos sensibles)                           │
│             │                                                          │
├─────────────┼──────────────────────────────────────────────────────────┤
│             │                                                          │
│ U2R         │ • su_attempted = 1 (intento escalada)                   │
│             │ • root_shell = 1 (acceso root)                          │
│             │ • num_shells > 0 (ejecución de comandos)               │
│             │ • num_access_files > 0 (acceso archivos)               │
│             │ • num_compromised > 0 (hosts comprometidos)             │
│             │                                                          │
├─────────────┼──────────────────────────────────────────────────────────┤
│             │                                                          │
│ Normal      │ • flag = SF (conexión completada normalmente)            │
│             │ • logged_in = 0 (no autentica)                          │
│             │ • num_failed_logins = 0 (sin fallos)                    │
│             │ • root_shell = 0 (sin acceso root)                      │
│             │ • duration ↔ (variable, servicio-dependiente)           │
│             │                                                          │
└─────────────┴──────────────────────────────────────────────────────────┘
```

---

## 🎯 CONCLUSIÓN

Las **41 características** fueron seleccionadas porque:

1. **Capturan características del protocolo**: `duration`, `protocol_type`, `service`, `flag`
2. **Miden volumen de tráfico**: `src_bytes`, `dst_bytes`
3. **Detectan anomalías obvias**: `land`, `wrong_fragment`, `urgent`
4. **Identifican intentos de acceso**: `num_failed_logins`, `logged_in`, `root_shell`
5. **Estadísticas temporales**: 8 features de `count`, `srv_count`, errores y tasas
6. **Análisis de comportamiento**: 9 features de estadísticas de host destino

**Cada característica fue diseñada por expertos en ciberseguridad** para detectar patrones específicos de ataques conocidos.

El modelo CNN/LSTM **aprende automáticamente** cuál es el peso de cada característica → cuáles son más importantes para clasificar → genera predicciones más precisas.

---

**Última actualización**: Noviembre 2025
