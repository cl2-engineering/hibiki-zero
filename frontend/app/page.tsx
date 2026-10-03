"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useAudioProcessor } from "./useAudioProcessor";
import { Circle, Download } from "lucide-react";
import { clsx } from "clsx";
import WaveformVisualizer from "../components/WaveformVisualizer";

export default function Home() {
  const [shouldConnect, setShouldConnect] = useState(false);
  // Mirrors shouldConnect so WebSocket callbacks can tell a server-side close
  // from one we asked for.
  const shouldConnectRef = useRef(false);
  const startingRef = useRef(false);
  // The server sends a handshake once it has a free slot for this connection.
  const [handshakeReceived, setHandshakeReceived] = useState(false);
  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();
  const [firstTime, setFirstTime] = useState(true);

  const [wordsReceived, setWordsReceived] = useState<
    { text: string; time: number }[]
  >([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [stepsSinceLastWord, setStepsSinceLastWord] = useState(0);

  // Messages are handled one at a time so the async Blob reads keep their order.
  const messageQueueRef = useRef<Promise<void>>(Promise.resolve());
  // useWebSocket needs the audio hook's callbacks and the audio hook needs
  // sendMessage, so the mic callback reaches sendMessage through this ref.
  const sendMessageRef = useRef<(message: Uint8Array) => void>(() => {});
  const handshakeReceivedRef = useRef(false);

  const wsProtocol =
    typeof window !== "undefined" && window.location.protocol === "https:"
      ? "wss:"
      : "ws:";
  // In development the Next.js server doesn't serve /api/chat, so
  // NEXT_PUBLIC_SERVER_HOST can point the page at the Python server instead.
  const serverHost =
    process.env.NEXT_PUBLIC_SERVER_HOST ||
    (typeof window !== "undefined" ? window.location.host : "localhost");
  const webSocketUrl = `${wsProtocol}//${serverHost}/api/chat`;

  const onAudioReceivedFromMic = useCallback((opusAudio: Uint8Array) => {
    // Until the handshake, the server isn't reading our audio yet, and anything
    // sent now would pile up and be translated late.
    if (!handshakeReceivedRef.current) return;
    const message = new Uint8Array(opusAudio.length + 1);
    message[0] = 1;
    message.set(opusAudio, 1);
    sendMessageRef.current(message);
  }, []);

  const {
    setupAudio,
    shutdownAudio,
    audioProcessor,
    processingDelaySec,
    hasRecording,
    getRecordingBlob,
  } = useAudioProcessor(onAudioReceivedFromMic);

  const stopSession = useCallback(
    (error?: string) => {
      shouldConnectRef.current = false;
      handshakeReceivedRef.current = false;
      setShouldConnect(false);
      setHandshakeReceived(false);
      shutdownAudio();
      if (error) setErrors((prev) => [...prev, error]);
    },
    [shutdownAudio],
  );

  const handleMessage = useCallback(
    async (data: unknown) => {
      if (!(data instanceof Blob)) {
        console.error("Expected Blob data, but received:", data);
        return;
      }
      const messageBytes = new Uint8Array(await data.arrayBuffer());
      const kind = messageBytes[0];
      const dataBytes = messageBytes.slice(1);

      if (kind === 0) {
        // Handshake
        handshakeReceivedRef.current = true;
        setHandshakeReceived(true);
      } else if (kind === 2) {
        // Text data
        const textDecoder = new TextDecoder();
        const text = textDecoder.decode(dataBytes);
        const TEXT_STREAM_OFFSET_MS = 160; // Hibiki's audio is delayed by two frames compared to the text.
        setWordsReceived((prev) => [
          ...prev,
          { text, time: Date.now() + TEXT_STREAM_OFFSET_MS },
        ]);
        setStepsSinceLastWord(0);
      } else if (kind === 1) {
        // Audio data
        const ap = audioProcessor.current;
        if (!ap) return;

        ap.decoder.postMessage({
          command: "decode",
          pages: dataBytes,
        });
        setStepsSinceLastWord((prev) => prev + 1);
      }
    },
    [audioProcessor],
  );

  const { sendMessage, readyState } = useWebSocket(
    webSocketUrl,
    {
      onMessage: (event) => {
        messageQueueRef.current = messageQueueRef.current
          .then(() => handleMessage(event.data))
          .catch((e) => console.error("Failed to handle message:", e));
      },
      onError: (event) => {
        console.error("WebSocket error:", event);
        if (!shouldConnectRef.current) return;
        stopSession(
          `Could not connect to the translation server at ${webSocketUrl}`,
        );
      },
      onClose: () => {
        if (!shouldConnectRef.current) return;
        stopSession("The translation server closed the connection.");
      },
    },
    shouldConnect,
  );

  useEffect(() => {
    sendMessageRef.current = sendMessage;
  }, [sendMessage]);

  // const connectionStatus = {
  //   [ReadyState.CONNECTING]: "Connecting",
  //   [ReadyState.OPEN]: "Connection open",
  //   [ReadyState.CLOSING]: "Connection closing",
  //   [ReadyState.CLOSED]: "Connection closed",
  //   [ReadyState.UNINSTANTIATED]: "Connection uninstantiated",
  // }[readyState];

  const onDownloadRecording = useCallback(() => {
    const blob = getRecordingBlob();
    const extension = blob.type.includes("mp4")
      ? "mp4"
      : blob.type.includes("ogg")
        ? "ogg"
        : "webm";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hibiki-zero-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [getRecordingBlob]);

  useEffect(() => {
    if (readyState === ReadyState.OPEN && shouldConnect) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFirstTime(false);
      setErrors([]);
    }
  }, [readyState, shouldConnect]);

  const onConnectButtonPress = async () => {
    // Ignore presses while a session is still starting
    if (startingRef.current) return;
    // If we're not connected yet
    if (!shouldConnect) {
      startingRef.current = true;
      const mediaStream = await askMicrophoneAccess();
      // If we have access to the microphone:
      const audioStarted =
        mediaStream !== null &&
        (await setupAudio(mediaStream).then(
          () => true,
          (e) => {
            console.error("Could not start audio:", e);
            mediaStream.getTracks().forEach((track) => track.stop());
            setErrors((prev) => [...prev, "Could not start audio playback."]);
            return false;
          },
        ));
      startingRef.current = false;
      if (audioStarted) {
        setWordsReceived([]);
        setStepsSinceLastWord(0);
        shouldConnectRef.current = true;
        setShouldConnect(true);
      }
    } else {
      stopSession();
      setErrors([]); // Clear previous connection errors
    }
  };

  const isTranslating = readyState === ReadyState.OPEN && handshakeReceived;
  const isWaitingForServer =
    readyState === ReadyState.OPEN && !handshakeReceived;

  const allErrors = errors.concat(
    microphoneAccess === "refused"
      ? [
          "Microphone access was refused. Please allow access and refresh the page.",
        ]
      : [],
    processingDelaySec > 0.5
      ? [
          `The model is ${processingDelaySec.toFixed(1)}s behind. Perhaps a network issue?`,
        ]
      : [],
  );

  return (
    <div className="flex min-h-screen justify-center bg-background text-textgray text-sm">
      <main className="flex min-h-screen w-xl max-w-screen flex-col items-center gap-4 py-10 px-4 bg-background sm:items-start">
        <h1 className="text-5xl text-green pb-1">Hibiki-Zero</h1>
        <div className="flex flex-col gap-2">
          <p>
            Kyutai&apos;s latest real-time speech-to-speech translation model.
            {/* TODO link to blog and code */}
          </p>
          <p>
            Hibiki-Zero translates into English from French, Spanish, German,
            and Portuguese.
          </p>
          <p>Use headphones for a better experience.</p>
        </div>

        <div className="w-full flex flex-row items-center justify-center gap-4">
          <button
            className={clsx(
              "flex flex-row items-center justify-between gap-2 cursor-pointer w-40 px-2 py-2",
              "text-xl",
              "border border-dashed",
              readyState === ReadyState.OPEN
                ? "border-white text-white"
                : "border-green text-green",
            )}
            onClick={() => onConnectButtonPress()}
          >
            <span>
              {isTranslating
                ? "Translating"
                : isWaitingForServer
                  ? "Waiting"
                  : "Translate"}
            </span>
            {readyState === ReadyState.OPEN && (
              <Circle
                size={24}
                color="var(--red)"
                fill="var(--red)"
                className="animate-pulse-recording"
              />
            )}
            {!(readyState === ReadyState.OPEN) && (
              <Circle size={24} color="var(--green)" />
            )}
          </button>
          {!shouldConnect && !firstTime && hasRecording && (
            <button
              className="flex flex-row items-center justify-center gap-2 cursor-pointer px-2 py-2 border border-dashed border-green text-green text-xl"
              onClick={onDownloadRecording}
            >
              <span>Download recording</span>
              <Download size={20} />
            </button>
          )}
        </div>
        {isWaitingForServer && (
          <p>
            The server is busy with another user, waiting for a free slot...
          </p>
        )}
        {allErrors.length > 0 && (
          <div>
            {allErrors.map((error, i) => (
              <p className="text-red" key={i}>
                {error}
              </p>
            ))}
          </div>
        )}
        {!firstTime && (
          <div className="w-full flex flex-col gap-4">
            <div className="relative flex flex-col">
              <span className="absolute top-2 left-2 text-textgray text-xs uppercase tracking-wider z-10 font-medium">
                You
              </span>
              <WaveformVisualizer
                width={800}
                height={120}
                waveformColor="#ffffff"
                textColor="#ffffff"
                displayDuration={4}
                analyzerNode={audioProcessor.current?.inputAnalyser || null}
                backgroundColor="transparent"
              />
            </div>
            <div className="relative flex flex-col">
              <span className="absolute top-2 left-2 text-textgray text-xs uppercase tracking-wider z-10 font-medium">
                Hibiki-Zero
              </span>
              <WaveformVisualizer
                width={800}
                height={120}
                waveformColor="#39F2AE"
                textColor="#39F2AE"
                displayDuration={4}
                analyzerNode={audioProcessor.current?.outputAnalyser || null}
                backgroundColor="transparent"
                textItems={wordsReceived}
              />
            </div>
          </div>
        )}
        {!firstTime && (
          <div className="bg-gray my-4 p-4 min-h-40 w-full">
            {isTranslating && wordsReceived.length === 0 ? (
              <span className="text-textgray">
                Speak to see your words translated...
              </span>
            ) : (
              <>
                <span>{wordsReceived.map((w) => w.text).join("")}</span>
              </>
            )}
            <span>
              {" "}
              {Array.from({
                length: Math.floor(stepsSinceLastWord / 25),
              }).map((_, i) => (
                <span className="text-textgray" key={i}>
                  &middot;{" "}
                </span>
              ))}
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
