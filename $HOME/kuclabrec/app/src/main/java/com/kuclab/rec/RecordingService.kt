package com.kuclab.rec

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import java.io.IOException

class RecordingService : Service() {

    private var recorder: MediaRecorder? = null
    private var currentFile: java.io.File? = null
    private val channelId = "rec_channel"

    companion object {
        const val ACTION_START = "com.kuclab.rec.START"
        const val ACTION_STOP = "com.kuclab.rec.STOP"
        const val EXTRA_NUMBER = "number"
        const val EXTRA_INCOMING = "incoming"
        var isRunning = false
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val number = intent.getStringExtra(EXTRA_NUMBER) ?: "unknown"
                val incoming = intent.getBooleanExtra(EXTRA_INCOMING, true)
                startRecording(number, incoming)
            }
            ACTION_STOP -> stopRecording()
        }
        return START_STICKY
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(NotificationManager::class.java)
            val chan = NotificationChannel(
                channelId,
                getString(R.string.app_name),
                NotificationManager.IMPORTANCE_LOW
            )
            mgr.createNotificationChannel(chan)
        }
    }

    private fun buildNotification(text: String): Notification {
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, channelId)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
    }

    private fun startRecording(number: String, incoming: Boolean) {
        if (recorder != null) return
        val file = RecorderStorage.newFile(this, number, incoming)
        currentFile = file
        val r = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(this)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }
        try {
            r.setAudioSource(MediaRecorder.AudioSource.VOICE_CALL)
        } catch (e: Exception) {
            try {
                r.setAudioSource(MediaRecorder.AudioSource.MIC)
            } catch (e2: Exception) {
                r.release()
                return
            }
        }
        r.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        r.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
        r.setOutputFile(file.absolutePath)
        try {
            r.prepare()
            r.start()
            recorder = r
            isRunning = true
            startForeground(1, buildNotification(getString(R.string.status_recording)))
        } catch (e: IOException) {
            r.release()
        }
    }

    private fun stopRecording() {
        try {
            recorder?.stop()
        } catch (e: Exception) {
        }
        recorder?.release()
        recorder = null
        isRunning = false
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        stopRecording()
        super.onDestroy()
    }
}
