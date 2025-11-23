const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const ROBOT_RADIUS = 20;
const SENSOR_LENGTH = 120;
const SENSOR_COUNT = 5;
const SENSOR_FOV = Math.PI / 1.5;

const ACTIONS = 4;
const MEMORY_SIZE = 2000;
const BATCH_SIZE = 64;
const DISCOUNT_FACTOR = 0.95;
const LEARNING_RATE = 0.001;
const EPSILON_DECAY = 0.995;
const MIN_EPSILON = 0.01;

const STEP_PENALTY = 0.01;
const PROGRESS_SCALE = 0.1;
const GOAL_REWARD = 10;
const COLLISION_PENALTY = -10;
const ORIENTATION_SCALE = 0.02;

const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('statusText');
const rewardText = document.getElementById('rewardText');
const episodeText = document.getElementById('episodeText');

canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;

let target = { x: 0, y: 0 };
let episode = 0;
let totalReward = 0;
let isTraining = true;
let epsilon = 1.0;

function dist(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

function randomizeTarget() {
    const margin = 50;
    target.x = Math.random() * (CANVAS_WIDTH - 2 * margin) + margin;
    target.y = Math.random() * (CANVAS_HEIGHT - 2 * margin) + margin;
}

class Brain {
    constructor() {
        this.memory = [];
        this.optimizer = tf.train.adam(LEARNING_RATE);
        this.model = this.createModel();
        this.model.compile({
            optimizer: this.optimizer,
            loss: 'meanSquaredError'
        });
    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [SENSOR_COUNT + 2], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: ACTIONS, activation: 'linear' }));
        return model;
    }

    predict(state) {
        return tf.tidy(() => {
            const xs = tf.tensor2d([state]);
            const ys = this.model.predict(xs);
            return ys.dataSync();
        });
    }

    act(state) {
        if (Math.random() < epsilon && isTraining) {
            return Math.floor(Math.random() * ACTIONS);
        }
        const qValues = this.predict(state);
        return qValues.indexOf(Math.max(...qValues));
    }

    remember(state, action, reward, nextState, done) {
        if (this.memory.length > MEMORY_SIZE) this.memory.shift();
        this.memory.push({ state, action, reward, nextState, done });
    }

    async replay() {
        if (this.memory.length < BATCH_SIZE) return;

        const batch = [];
        for (let i = 0; i < BATCH_SIZE; i++) {
            batch.push(this.memory[Math.floor(Math.random() * this.memory.length)]);
        }

        const states = batch.map(e => e.state);
        const nextStates = batch.map(e => e.nextState);

        const qValues = tf.tidy(() => {
            const xs = tf.tensor2d(states);
            return this.model.predict(xs).arraySync();
        });

        const qNextValues = tf.tidy(() => {
            const xs = tf.tensor2d(nextStates);
            return this.model.predict(xs).arraySync();
        });

        const x = [];
        const y = [];

        for (let i = 0; i < BATCH_SIZE; i++) {
            const { state, action, reward, done } = batch[i];
            let targetQ = qValues[i].slice();

            if (done) {
                targetQ[action] = reward;
            } else {
                targetQ[action] = reward + DISCOUNT_FACTOR * Math.max(...qNextValues[i]);
            }

            x.push(state);
            y.push(targetQ);
        }

        const xs = tf.tensor2d(x);
        const ys = tf.tensor2d(y);

        await this.model.fit(xs, ys, { epochs: 1, verbose: 0 });

        xs.dispose();
        ys.dispose();

        if (epsilon > MIN_EPSILON) epsilon *= EPSILON_DECAY;
    }

    async save(filename) {
        await this.model.save(`downloads://${filename}`);
        console.log('Modelo descargado');
    }

    async load(files) {
        try {
            const model = await tf.loadLayersModel(tf.io.browserFiles(files));
            this.model.dispose();
            this.model = model;
            this.model.compile({ optimizer: this.optimizer, loss: 'meanSquaredError' });
            console.log('Modelo cargado desde archivos');
            return true;
        } catch (e) {
            console.error('No se pudo cargar el modelo', e);
            return false;
        }
    }
}

class Robot {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.angle = -Math.PI / 2;
        this.velocity = 0;
        this.rotation = 0;

        this.sensors = [];
        this.sensorReadings = new Array(SENSOR_COUNT).fill(1);

        this.brain = new Brain();
        this.alive = true;

        this.lastDist = dist(this.x, this.y, target.x, target.y);
    }

    getState() {
        const d = dist(this.x, this.y, target.x, target.y) / CANVAS_WIDTH;
        let angleToTarget = Math.atan2(target.y - this.y, target.x - this.x) - this.angle;
        while (angleToTarget > Math.PI) angleToTarget -= 2 * Math.PI;
        while (angleToTarget < -Math.PI) angleToTarget += 2 * Math.PI;
        const a = angleToTarget / Math.PI;

        return [...this.sensorReadings, d, a];
    }

    update() {
        if (!this.alive) return;

        const currentState = this.getState();
        const action = this.brain.act(currentState);

        if (action === 0) {
            this.velocity = 3;
            this.rotation = 0;
        } else if (action === 1) {
            this.velocity = 1.5;
            this.rotation = -0.1;
        } else if (action === 2) {
            this.velocity = 1.5;
            this.rotation = 0.1;
        } else if (action === 3) {
            this.velocity = -2;
            this.rotation = 0;
        }

        this.x += Math.cos(this.angle) * this.velocity;
        this.y += Math.sin(this.angle) * this.velocity;
        this.angle += this.rotation;

        this.updateSensors();

        let reward = 0;
        let done = false;

        const d = dist(this.x, this.y, target.x, target.y);

        let angleToTarget = Math.atan2(target.y - this.y, target.x - this.x) - this.angle;
        while (angleToTarget > Math.PI) angleToTarget -= 2 * Math.PI;
        while (angleToTarget < -Math.PI) angleToTarget += 2 * Math.PI;
        const cosAngle = Math.cos(angleToTarget);

        const progress = this.lastDist - d;
        reward += progress * PROGRESS_SCALE;
        this.lastDist = d;

        reward -= STEP_PENALTY;

        if (this.velocity > 0) {
            reward += ORIENTATION_SCALE * cosAngle;
        } else if (this.velocity < 0) {
            reward -= ORIENTATION_SCALE * cosAngle;
        }

        const hitWall = (
            this.x < ROBOT_RADIUS ||
            this.x > CANVAS_WIDTH - ROBOT_RADIUS ||
            this.y < ROBOT_RADIUS ||
            this.y > CANVAS_HEIGHT - ROBOT_RADIUS ||
            this.sensorReadings.some(r => r < 0.1)
        );

        if (hitWall) {
            reward = COLLISION_PENALTY;
            done = true;
            this.alive = false;
        }

        if (!done && d < 30) {
            reward += GOAL_REWARD;
            done = true;
            this.alive = false;
        }

        if (isTraining) {
            const nextState = this.getState();
            this.brain.remember(currentState, action, reward, nextState, done);
            this.brain.replay();
        }

        updateReward(reward);

        if (done) {
            this.reset();
        }
    }

    updateSensors() {
        this.sensors = [];
        for (let i = 0; i < SENSOR_COUNT; i++) {
            let sensorAngle = this.angle - (SENSOR_FOV / 2) + (SENSOR_FOV / (SENSOR_COUNT - 1)) * i;
            let startX = this.x + Math.cos(this.angle) * ROBOT_RADIUS;
            let startY = this.y + Math.sin(this.angle) * ROBOT_RADIUS;
            let endX = startX + Math.cos(sensorAngle) * SENSOR_LENGTH;
            let endY = startY + Math.sin(sensorAngle) * SENSOR_LENGTH;

            let reading = 1;
            let intersection = this.getIntersection(startX, startY, endX, endY);
            if (intersection) {
                endX = intersection.x;
                endY = intersection.y;
                reading = dist(startX, startY, endX, endY) / SENSOR_LENGTH;
            }
            this.sensors.push({ x1: startX, y1: startY, x2: endX, y2: endY });
            this.sensorReadings[i] = reading;
        }
    }

    getIntersection(x1, y1, x2, y2) {
        const walls = [
            { x1: 0, y1: 0, x2: CANVAS_WIDTH, y2: 0 },
            { x1: CANVAS_WIDTH, y1: 0, x2: CANVAS_WIDTH, y2: CANVAS_HEIGHT },
            { x1: CANVAS_WIDTH, y1: CANVAS_HEIGHT, x2: 0, y2: CANVAS_HEIGHT },
            { x1: 0, y1: CANVAS_HEIGHT, x2: 0, y2: 0 }
        ];
        let closest = null;
        let minD = Infinity;
        for (let wall of walls) {
            const pt = this.cast(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2);
            if (pt) {
                const d = dist(x1, y1, pt.x, pt.y);
                if (d < minD) { minD = d; closest = pt; }
            }
        }
        return closest;
    }

    cast(x1, y1, x2, y2, x3, y3, x4, y4) {
        const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (den == 0) return null;
        const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
        const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;
        if (t > 0 && t < 1 && u > 0 && u < 1) {
            return { x: x1 + t * (x2 - x1), y: y1 + t * (y2 - y1) };
        }
        return null;
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);

        ctx.beginPath();
        ctx.arc(0, 0, ROBOT_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#1e3a8a';
        ctx.fill();
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#333';
        ctx.fillRect(-ROBOT_RADIUS, -ROBOT_RADIUS - 5, 10, 5);
        ctx.fillRect(-ROBOT_RADIUS, ROBOT_RADIUS, 10, 5);

        ctx.beginPath();
        ctx.moveTo(ROBOT_RADIUS - 5, 0);
        ctx.lineTo(ROBOT_RADIUS + 10, 0);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.restore();

        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(target.x, target.y);
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.3)';
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);

        for (let s of this.sensors) {
            ctx.beginPath();
            ctx.moveTo(s.x1, s.y1);
            ctx.lineTo(s.x2, s.y2);
            ctx.strokeStyle = `rgba(255, ${Math.floor(255 * (dist(s.x1, s.y1, s.x2, s.y2) / SENSOR_LENGTH))}, 0, 0.5)`;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(s.x2, s.y2, 2, 0, Math.PI * 2);
            ctx.fillStyle = 'yellow';
            ctx.fill();
        }
    }

    reset() {
        this.x = CANVAS_WIDTH / 2;
        this.y = CANVAS_HEIGHT / 2;
        this.angle = -Math.PI / 2;
        this.velocity = 0;
        this.rotation = 0;
        this.alive = true;

        episode++;
        episodeText.innerText = episode;
        totalReward = 0;
        statusText.innerText = `Epsilon: ${epsilon.toFixed(3)}`;

        randomizeTarget();
        this.lastDist = dist(this.x, this.y, target.x, target.y);
    }
}

randomizeTarget();
const robot = new Robot(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);

canvas.addEventListener('mousedown', e => {
    const rect = canvas.getBoundingClientRect();
    target.x = e.clientX - rect.left;
    target.y = e.clientY - rect.top;
    robot.lastDist = dist(robot.x, robot.y, target.x, target.y);
});

document.getElementById('resetBtn').addEventListener('click', () => robot.reset());

document.getElementById('toggleTrainingBtn').addEventListener('click', () => {
    isTraining = !isTraining;
    document.getElementById('toggleTrainingBtn').innerText =
        isTraining ? "Pausar Entrenamiento" : "Reanudar Entrenamiento";
});

document.getElementById('saveBtn').addEventListener('click', async () => {
    const filename = `rl-robot-model-episode-${episode}`;
    await robot.brain.save(filename);
});

document.getElementById('loadBtn').addEventListener('click', () => {
    document.getElementById('modelInput').click();
});

document.getElementById('modelInput').addEventListener('change', async (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        const success = await robot.brain.load(files);
        if (success) {
            alert('Modelo cargado exitosamente. Epsilon reseteado a 0.1.');
            epsilon = 0.1;
            robot.reset();
        } else {
            alert('Error al cargar. Asegúrate de seleccionar AMBOS archivos (.json y .bin).');
        }
    }
});

function updateReward(val) {
    totalReward += val;
    rewardText.innerText = totalReward.toFixed(1);
}

function drawTarget() {
    ctx.beginPath();
    ctx.arc(target.x, target.y, 10, 0, Math.PI * 2);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#22c55e';
}

function loop() {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    drawTarget();
    robot.update();
    robot.draw();

    requestAnimationFrame(loop);
}

loop();
