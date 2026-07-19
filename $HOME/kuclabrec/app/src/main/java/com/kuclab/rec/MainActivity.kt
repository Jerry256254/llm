package com.kuclab.rec

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.res.stringResource
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {

    private var callListener: CallStateListener? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentSafe()
    }

    private fun setContentSafe() {
        val telephony = getSystemService(Context.TELEPHONY_SERVICE) as android.telephony.TelephonyManager
        callListener = CallStateListener(this)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            @Suppress("DEPRECATION")
            telephony.listen(callListener, android.telephony.PhoneStateListener.LISTEN_CALL_STATE)
        }

        setContent {
            AppRoot(
                onRegisterListener = {
                    if (ContextCompat.checkSelfPermission(
                            this, Manifest.permission.READ_PHONE_STATE
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        @Suppress("DEPRECATION")
                        telephony.listen(
                            callListener,
                            android.telephony.PhoneStateListener.LISTEN_CALL_STATE
                        )
                    }
                }
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        val telephony = getSystemService(Context.TELEPHONY_SERVICE) as android.telephony.TelephonyManager
        @Suppress("DEPRECATION")
        telephony.listen(callListener, android.telephony.PhoneStateListener.LISTEN_NONE)
    }
}

@Composable
private fun AppRoot(onRegisterListener: () -> Unit) {
    MaterialTheme {
        val context = LocalContext.current
        var recordings by remember { mutableStateOf(RecorderStorage.listRecordings(context)) }
        var serviceOn by remember { mutableStateOf(RecordingService.isRunning) }

        LaunchedEffect(Unit) {
            recordings = RecorderStorage.listRecordings(context)
        }

        val perms = remember {
            mutableListOf(
                Manifest.permission.READ_PHONE_STATE,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.READ_CONTACTS
            ).apply {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    add(Manifest.permission.POST_NOTIFICATIONS)
                }
            }
        }

        val launcher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { result ->
            if (result.values.all { it }) onRegisterListener()
        }

        MainScreen(
            serviceOn = serviceOn,
            recordings = recordings,
            hasPerms = perms.all {
                ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
            },
            onToggle = {
                if (serviceOn) {
                    context.startService(Intent(context, RecordingService::class.java).apply {
                        action = RecordingService.ACTION_STOP
                    })
                    serviceOn = false
                } else {
                    val needed = perms.filter {
                        ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
                    }
                    if (needed.isNotEmpty()) {
                        launcher.launch(needed.toTypedArray())
                    }
                    context.startForegroundService(Intent(context, RecordingService::class.java).apply {
                        action = RecordingService.ACTION_START
                        putExtra(RecordingService.EXTRA_NUMBER, "init")
                        putExtra(RecordingService.EXTRA_INCOMING, true)
                    })
                    serviceOn = true
                    onRegisterListener()
                }
            },
            onRequestPerms = { launcher.launch(perms.toTypedArray()) },
            onBattery = {
                val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
                if (!pm.isIgnoringBatteryOptimizations(context.packageName)) {
                    context.startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                        data = Uri.parse("package:${context.packageName}")
                    })
                }
            },
            onRefresh = { recordings = RecorderStorage.listRecordings(context) },
            onPlay = { rec ->
                val i = Intent(Intent.ACTION_VIEW).apply {
                    setDataAndType(Uri.fromFile(rec.file), "audio/*")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(i)
            },
            onDelete = { rec ->
                rec.file.delete()
                recordings = RecorderStorage.listRecordings(context)
            }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    serviceOn: Boolean,
    recordings: List<Recording>,
    hasPerms: Boolean,
    onToggle: () -> Unit,
    onRequestPerms: () -> Unit,
    onBattery: () -> Unit,
    onRefresh: () -> Unit,
    onPlay: (Recording) -> Unit,
    onDelete: (Recording) -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(title = { Text("KucLab Rec") })
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text(
                            text = if (serviceOn)
                                stringResource(R.string.service_running)
                            else stringResource(R.string.service_stopped),
                            style = MaterialTheme.typography.titleMedium
                        )
                        Text(
                            text = if (serviceOn) "Recording ON" else "Recording OFF",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                    Switch(checked = serviceOn, onCheckedChange = { onToggle() })
                }
            }

            if (!hasPerms) {
                Button(onClick = onRequestPerms, modifier = Modifier.fillMaxWidth()) {
                    Text("Udělit oprávnění / Grant permissions")
                }
            }

            Button(onClick = onBattery, modifier = Modifier.fillMaxWidth()) {
                Text("Vypnout optimalizaci baterie / Disable battery optimization")
            }

            Text(
                text = "Nahrávky / Recordings",
                style = MaterialTheme.typography.titleMedium
            )

            if (recordings.isEmpty()) {
                Text("Zatím žádné nahrávky / No recordings yet")
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(recordings) { rec ->
                        RecordingItem(rec, onPlay, onDelete)
                    }
                }
            }
        }
    }
}

@Composable
fun RecordingItem(
    rec: Recording,
    onPlay: (Recording) -> Unit,
    onDelete: (Recording) -> Unit
) {
    val fmt = SimpleDateFormat("dd.MM.yyyy HH:mm", Locale.getDefault())
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onPlay(rec) }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = rec.name, style = MaterialTheme.typography.titleSmall)
                Text(text = rec.number, style = MaterialTheme.typography.bodySmall)
                Text(
                    text = "${fmt.format(Date(rec.date))} • ${if (rec.incoming) "Příchozí" else "Odchozí"}",
                    style = MaterialTheme.typography.bodySmall
                )
            }
            Row {
                IconButton(onClick = { onPlay(rec) }) {
                    Icon(Icons.Default.PlayArrow, contentDescription = "Play")
                }
                IconButton(onClick = { onDelete(rec) }) {
                    Icon(Icons.Default.Delete, contentDescription = "Delete")
                }
            }
        }
    }
}
