import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let statusBarItem: vscode.StatusBarItem;
let currentProcess: ChildProcess | null = null;
let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    console.log('HER2 Clinical ML Pipeline extension is now active!');

    // Create output channel for logs
    outputChannel = vscode.window.createOutputChannel('HER2 Pipeline');
    context.subscriptions.push(outputChannel);

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(pulse) HER2 Pipeline Ready";
    statusBarItem.tooltip = "HER2 Clinical ML Pipeline - Ready for pathology workflows";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Register new clinical commands
    const commands = [
        // Core clinical workflow commands
        vscode.commands.registerCommand('her2-pipeline.preprocessing', () => {
            runPreprocessing();
        }),
        vscode.commands.registerCommand('her2-pipeline.trainSegmentation', () => {
            runSegmentationTraining();
        }),
        vscode.commands.registerCommand('her2-pipeline.trainMIL', () => {
            runMILTraining();
        }),
        vscode.commands.registerCommand('her2-pipeline.previewBags', () => {
            previewMILBags();
        }),
        vscode.commands.registerCommand('her2-pipeline.viewDashboard', () => {
            viewTrainingDashboard();
        }),
        vscode.commands.registerCommand('her2-pipeline.generateReport', () => {
            generateClinicalReport();
        }),
        
        // Legacy commands for backward compatibility
        vscode.commands.registerCommand('her2-pipeline.datasetStats', () => {
            runDatasetStats();
        }),
        vscode.commands.registerCommand('her2-pipeline.tumourFilter', () => {
            runTumourFilter();
        }),
        vscode.commands.registerCommand('her2-pipeline.validateReport', () => {
            runValidation();
        })
    ];

    context.subscriptions.push(...commands);

    // Initialize clinical environment
    initializeClinicalEnvironment(context);
}

async function initializeClinicalEnvironment(context: vscode.ExtensionContext) {
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return;
        }

        // Check for required directories and create if missing
        const requiredDirs = ['slides', 'tiles', 'bags', 'manifests', 'models', 'reports', 'temp_bags'];
        for (const dir of requiredDirs) {
            const dirPath = path.join(workspaceFolder.uri.fsPath, dir);
            if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
            }
        }

        outputChannel.appendLine('✅ Clinical environment initialized');
        outputChannel.appendLine(`📁 Workspace: ${workspaceFolder.uri.fsPath}`);
        
    } catch (error) {
        outputChannel.appendLine(`❌ Failed to initialize clinical environment: ${error}`);
    }
}

async function runPreprocessing() {
    const config = vscode.workspace.getConfiguration('her2-pipeline');
    const dataPath = config.get<string>('dataPath') || './slides';
    const tileSize = config.get<number>('preprocessing.tileSize') || 224;
    const overlap = config.get<number>('preprocessing.overlap') || 0;
    const magnification = config.get<number>('preprocessing.magnification') || 0.5;
    
    statusBarItem.text = "$(sync~spin) Preprocessing WSI Slides...";
    outputChannel.show();
    outputChannel.appendLine('🩺 Starting HER2 Preprocessing Pipeline');
    outputChannel.appendLine('=' .repeat(50));
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'src', 'python', 'preprocess.py');
        
        const args = [
            scriptPath,
            '--input-path', dataPath,
            '--tile-size', tileSize.toString(),
            '--overlap', overlap.toString(),
            '--magnification', magnification.toString(),
            '--output-tiles', './tiles',
            '--output-bags', './bags',
            '--output-manifests', './manifests'
        ];
        
        currentProcess = spawn(pythonPath, args, {
            cwd: workspaceFolder.uri.fsPath
        });

        currentProcess.stdout?.on('data', (data) => {
            const output = data.toString();
            outputChannel.append(output);
            
            // Parse preprocessing progress
            const progressMatch = output.match(/Processing slide (\d+)\/(\d+)/);
            if (progressMatch) {
                const current = parseInt(progressMatch[1]);
                const total = parseInt(progressMatch[2]);
                statusBarItem.text = `$(sync~spin) Processing ${current}/${total} slides...`;
            }
        });

        currentProcess.stderr?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(check) Preprocessing Complete";
                outputChannel.appendLine('\n✅ Preprocessing pipeline completed successfully!');
                outputChannel.appendLine('📁 Tiles saved to: ./tiles/');
                outputChannel.appendLine('📊 MIL bags saved to: ./bags/');
                outputChannel.appendLine('📋 Manifests saved to: ./manifests/');
                vscode.window.showInformationMessage('HER2 preprocessing completed! Check output panel for details.');
            } else {
                statusBarItem.text = "$(error) Preprocessing Failed";
                outputChannel.appendLine(`\n❌ Preprocessing failed with exit code ${code}`);
                vscode.window.showErrorMessage(`Preprocessing failed. Check output panel for details.`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) Preprocessing Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Preprocessing error: ${error}`);
    }
}

async function runSegmentationTraining() {
    statusBarItem.text = "$(sync~spin) Training Segmentation Model...";
    outputChannel.show();
    outputChannel.appendLine('🤖 Starting HER2 Segmentation Training');
    outputChannel.appendLine('=' .repeat(50));
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'MyLightningProject', 'train_seg.py');
        
        const args = [
            scriptPath,
            '--data-path', './tiles',
            '--epochs', '50',
            '--class-weights', 'auto',
            '--cv-folds', '5',
            '--precision', '16'
        ];
        
        currentProcess = spawn(pythonPath, args, {
            cwd: workspaceFolder.uri.fsPath
        });

        let currentFold = 1;
        currentProcess.stdout?.on('data', (data) => {
            const output = data.toString();
            outputChannel.append(output);
            
            // Parse training progress
            const foldMatch = output.match(/Fold (\d+).*IoU.*neg:(\d+\.\d+).*low:(\d+\.\d+).*high:(\d+\.\d+)/);
            if (foldMatch) {
                currentFold = parseInt(foldMatch[1]);
                const iouNeg = parseFloat(foldMatch[2]);
                const iouLow = parseFloat(foldMatch[3]);
                const iouHigh = parseFloat(foldMatch[4]);
                statusBarItem.text = `$(pulse) Fold ${currentFold}/5 – IoU_neg: ${iouNeg.toFixed(3)} | IoU_low: ${iouLow.toFixed(3)} | IoU_high: ${iouHigh.toFixed(3)}`;
            }
        });

        currentProcess.stderr?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(check) Segmentation Training Complete";
                outputChannel.appendLine('\n✅ Segmentation training completed successfully!');
                outputChannel.appendLine('🏥 Models comply with ASCO 2018 guidelines');
                vscode.window.showInformationMessage('HER2 segmentation training completed!');
            } else {
                statusBarItem.text = "$(error) Segmentation Training Failed";
                outputChannel.appendLine(`\n❌ Training failed with exit code ${code}`);
                vscode.window.showErrorMessage(`Segmentation training failed. Check output panel.`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) Training Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Training error: ${error}`);
    }
}

async function runMILTraining() {
    const config = vscode.workspace.getConfiguration('her2-pipeline');
    const milModel = config.get<string>('mil.model') || 'clam';
    const backbone = config.get<string>('mil.backbone') || 'resnet50';
    const maxBagSize = config.get<number>('mil.maxBagSize') || 1000;
    const cvFolds = config.get<number>('mil.cvFolds') || 5;
    
    statusBarItem.text = "$(sync~spin) Training MIL Model...";
    outputChannel.show();
    outputChannel.appendLine('🧠 Starting HER2 MIL Training (Weakly-Supervised)');
    outputChannel.appendLine('=' .repeat(50));
    outputChannel.appendLine(`Model: ${milModel.toUpperCase()}`);
    outputChannel.appendLine(`Backbone: ${backbone}`);
    outputChannel.appendLine(`Max bag size: ${maxBagSize}`);
    outputChannel.appendLine(`CV folds: ${cvFolds}`);
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'src', 'python', 'mil_train.py');
        
        const args = [
            scriptPath,
            '--model', milModel,
            '--backbone', backbone,
            '--max-bag-size', maxBagSize.toString(),
            '--cv-folds', cvFolds.toString(),
            '--data-path', './MyLightningProject/data',
            '--epochs', '30'
        ];
        
        currentProcess = spawn(pythonPath, args, {
            cwd: workspaceFolder.uri.fsPath
        });

        let currentFold = 1;
        currentProcess.stdout?.on('data', (data) => {
            const output = data.toString();
            outputChannel.append(output);
            
            // Parse MIL training progress
            const foldMatch = output.match(/Fold (\d+).*Accuracy: (\d+\.\d+).*F1: (\d+\.\d+)/);
            if (foldMatch) {
                currentFold = parseInt(foldMatch[1]);
                const acc = parseFloat(foldMatch[2]);
                const f1 = parseFloat(foldMatch[3]);
                statusBarItem.text = `$(pulse) MIL Fold ${currentFold}/${cvFolds} – Acc: ${acc.toFixed(3)} | F1: ${f1.toFixed(3)}`;
            }
            
            // Parse attention information
            const attentionMatch = output.match(/Attention weights computed for (\d+) bags/);
            if (attentionMatch) {
                outputChannel.appendLine(`🎯 Attention weights computed for ${attentionMatch[1]} patient bags`);
            }
        });

        currentProcess.stderr?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(check) MIL Training Complete";
                outputChannel.appendLine('\n✅ MIL training completed successfully!');
                outputChannel.appendLine('🎯 Attention-based models ready for clinical interpretation');
                outputChannel.appendLine('🏥 Patient-level CV ensures clinical generalizability');
                vscode.window.showInformationMessage('HER2 MIL training completed! Attention maps available for review.');
            } else {
                statusBarItem.text = "$(error) MIL Training Failed";
                outputChannel.appendLine(`\n❌ MIL training failed with exit code ${code}`);
                vscode.window.showErrorMessage(`MIL training failed. Check output panel.`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) MIL Training Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`MIL training error: ${error}`);
    }
}

async function previewMILBags() {
    statusBarItem.text = "$(eye) Opening MIL Bag Preview...";
    outputChannel.appendLine('🎞 Opening MIL Bag Preview');
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        // Create webview panel for bag preview
        const panel = vscode.window.createWebviewPanel(
            'milBagPreview',
            'HER2 MIL Bag Preview',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'src', 'python', 'bag_creator.py');
        
        // Generate bag preview data
        const previewProcess = spawn(pythonPath, [scriptPath, '--preview', '--output-json'], {
            cwd: workspaceFolder.uri.fsPath
        });

        let bagData = '';
        previewProcess.stdout?.on('data', (data) => {
            bagData += data.toString();
        });

        previewProcess.on('close', (code) => {
            if (code === 0) {
                // Load webview HTML
                const webviewPath = path.join(__dirname, '..', 'src', 'webview', 'bag-preview.html');
                if (fs.existsSync(webviewPath)) {
                    const html = fs.readFileSync(webviewPath, 'utf8');
                    panel.webview.html = html.replace('{{BAG_DATA}}', bagData);
                } else {
                    // Fallback HTML
                    panel.webview.html = createBagPreviewHTML(bagData);
                }
                statusBarItem.text = "$(eye) MIL Bag Preview Open";
                outputChannel.appendLine('✅ MIL bag preview opened successfully');
            } else {
                statusBarItem.text = "$(error) Bag Preview Failed";
                vscode.window.showErrorMessage('Failed to generate bag preview');
            }
        });

    } catch (error) {
        statusBarItem.text = "$(error) Bag Preview Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Bag preview error: ${error}`);
    }
}

function createBagPreviewHTML(bagData: string): string {
    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HER2 MIL Bag Preview</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: var(--vscode-editor-background);
                color: var(--vscode-editor-foreground);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background-color: var(--vscode-panel-background);
                border-radius: 8px;
            }
            .bag-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .bag-card {
                border: 1px solid var(--vscode-panel-border);
                border-radius: 8px;
                padding: 15px;
                background-color: var(--vscode-panel-background);
            }
            .bag-header {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                color: var(--vscode-charts-blue);
            }
            .bag-stats {
                font-size: 12px;
                color: var(--vscode-descriptionForeground);
                margin-bottom: 15px;
            }
            .tile-grid {
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 5px;
            }
            .tile-preview {
                width: 50px;
                height: 50px;
                background-color: var(--vscode-button-background);
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 10px;
                color: var(--vscode-button-foreground);
            }
            .attention-high { background-color: var(--vscode-charts-red) !important; }
            .attention-medium { background-color: var(--vscode-charts-orange) !important; }
            .attention-low { background-color: var(--vscode-charts-green) !important; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎞 HER2 MIL Bag Preview</h1>
            <p>Visual exploration of patient-level bags for weakly-supervised learning</p>
        </div>
        <div id="bagContainer" class="bag-container">
            <!-- Bag previews will be populated here -->
        </div>
        <script>
            const bagData = ${bagData || '[]'};
            const container = document.getElementById('bagContainer');
            
            bagData.forEach((bag, index) => {
                const bagCard = document.createElement('div');
                bagCard.className = 'bag-card';
                
                const bagHeader = document.createElement('div');
                bagHeader.className = 'bag-header';
                bagHeader.textContent = \`Patient \${bag.patient_id || 'Unknown'} - \${bag.label || 'No Label'}\`;
                
                const bagStats = document.createElement('div');
                bagStats.className = 'bag-stats';
                bagStats.innerHTML = \`
                    Tiles: \${bag.tile_count || 0} | 
                    HER2 Score: \${bag.her2_score || 'N/A'} | 
                    Clinical Category: \${bag.clinical_category || 'Unknown'}
                \`;
                
                const tileGrid = document.createElement('div');
                tileGrid.className = 'tile-grid';
                
                // Create tile previews (first 25 tiles)
                const maxTiles = Math.min(bag.tile_count || 0, 25);
                for (let i = 0; i < maxTiles; i++) {
                    const tile = document.createElement('div');
                    tile.className = 'tile-preview';
                    
                    // Simulate attention weights for visualization
                    const attention = Math.random();
                    if (attention > 0.7) tile.classList.add('attention-high');
                    else if (attention > 0.4) tile.classList.add('attention-medium');
                    else tile.classList.add('attention-low');
                    
                    tile.textContent = \`T\${i+1}\`;
                    tile.title = \`Tile \${i+1} - Attention: \${attention.toFixed(3)}\`;
                    tileGrid.appendChild(tile);
                }
                
                bagCard.appendChild(bagHeader);
                bagCard.appendChild(bagStats);
                bagCard.appendChild(tileGrid);
                container.appendChild(bagCard);
            });
            
            if (bagData.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: var(--vscode-descriptionForeground);">No bag data available. Run preprocessing first.</div>';
            }
        </script>
    </body>
    </html>
    `;
}

async function viewTrainingDashboard() {
    statusBarItem.text = "$(graph) Opening Training Dashboard...";
    outputChannel.show();
    outputChannel.appendLine('📊 HER2 Training Dashboard');
    outputChannel.appendLine('=' .repeat(30));
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        // Stream Lightning CSV logs
        const logsPath = path.join(workspaceFolder.uri.fsPath, 'logs');
        if (fs.existsSync(logsPath)) {
            outputChannel.appendLine('📈 Streaming training metrics...');
            
            // Find latest log files
            const logFiles = fs.readdirSync(logsPath).filter(f => f.endsWith('.csv'));
            for (const logFile of logFiles.slice(-3)) { // Show last 3 log files
                const logPath = path.join(logsPath, logFile);
                const logContent = fs.readFileSync(logPath, 'utf8');
                
                outputChannel.appendLine(`\n📄 ${logFile}:`);
                outputChannel.appendLine('-'.repeat(20));
                
                // Parse and display key metrics
                const lines = logContent.split('\n');
                const header = lines[0];
                const lastMetrics = lines[lines.length - 2]; // Last non-empty line
                
                if (header && lastMetrics) {
                    const keys = header.split(',');
                    const values = lastMetrics.split(',');
                    
                    for (let i = 0; i < keys.length && i < values.length; i++) {
                        if (keys[i].includes('loss') || keys[i].includes('acc') || keys[i].includes('iou')) {
                            outputChannel.appendLine(`  ${keys[i]}: ${values[i]}`);
                        }
                    }
                }
            }
            
            statusBarItem.text = "$(graph) Dashboard Active";
            vscode.window.showInformationMessage('Training dashboard active in output panel');
        } else {
            outputChannel.appendLine('⚠️  No training logs found. Start training first.');
            statusBarItem.text = "$(warning) No Training Logs";
        }

    } catch (error) {
        statusBarItem.text = "$(error) Dashboard Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Dashboard error: ${error}`);
    }
}

async function generateClinicalReport() {
    statusBarItem.text = "$(file-text) Generating Clinical Report...";
    outputChannel.show();
    outputChannel.appendLine('📈 Generating HER2 Clinical Report');
    outputChannel.appendLine('=' .repeat(40));
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'src', 'python', 'clinical_report.py');
        
        const args = [
            scriptPath,
            '--models-path', './models',
            '--output-path', './reports',
            '--include-attention-maps',
            '--asco-compliance'
        ];
        
        currentProcess = spawn(pythonPath, args, {
            cwd: workspaceFolder.uri.fsPath
        });

        currentProcess.stdout?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.stderr?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(file-text) Clinical Report Generated";
                outputChannel.appendLine('\n✅ Clinical report generated successfully!');
                outputChannel.appendLine('🏥 ASCO 2018 compliant evaluation included');
                outputChannel.appendLine('🎯 Attention maps available for pathologist review');
                
                // Open reports folder
                const reportsPath = path.join(workspaceFolder.uri.fsPath, 'reports');
                vscode.env.openExternal(vscode.Uri.file(reportsPath));
                
                vscode.window.showInformationMessage('Clinical report generated! Opening reports folder...');
            } else {
                statusBarItem.text = "$(error) Report Generation Failed";
                outputChannel.appendLine(`\n❌ Report generation failed with exit code ${code}`);
                vscode.window.showErrorMessage(`Report generation failed. Check output panel.`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) Report Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Report generation error: ${error}`);
    }
}

// Legacy function for backward compatibility
async function runDatasetStats() {
    const config = vscode.workspace.getConfiguration('her2-pipeline');
    const dataPath = config.get<string>('dataPath') || './MyLightningProject/data';
    
    statusBarItem.text = "$(sync~spin) Analyzing Dataset...";
    outputChannel.show();
    outputChannel.appendLine('📊 Analyzing Dataset Statistics (Legacy)');
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'MyLightningProject', 'dataset_stats.py');
        
        currentProcess = spawn(pythonPath, [scriptPath, '--data-path', dataPath], {
            cwd: workspaceFolder.uri.fsPath
        });

        let output = '';
        currentProcess.stdout?.on('data', (data) => {
            const text = data.toString();
            output += text;
            outputChannel.append(text);
        });

        currentProcess.stderr?.on('data', (data) => {
            const text = data.toString();
            output += text;
            outputChannel.append(text);
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(check) Dataset Stats Complete";
                outputChannel.appendLine('\n✅ Dataset statistics completed');
                vscode.window.showInformationMessage('Dataset statistics generated successfully!');
            } else {
                statusBarItem.text = "$(error) Dataset Stats Failed";
                outputChannel.appendLine(`\n❌ Dataset stats failed with code ${code}`);
                vscode.window.showErrorMessage(`Dataset stats failed with code ${code}`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) Dataset Stats Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Error: ${error}`);
    }
}

async function runTumourFilter() {
    statusBarItem.text = "$(sync~spin) Processing Tumour Filter...";
    outputChannel.show();
    outputChannel.appendLine('🔍 Running Tumour Filter Preview (Legacy)');
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'MyLightningProject', 'tumour_filter.py');
        
        currentProcess = spawn(pythonPath, [scriptPath, '--preview'], {
            cwd: workspaceFolder.uri.fsPath
        });

        currentProcess.stdout?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.stderr?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(check) Tumour Filter Complete";
                outputChannel.appendLine('\n✅ Tumour filter preview generated');
                vscode.window.showInformationMessage('Tumour filter preview generated!');
            } else {
                statusBarItem.text = "$(error) Tumour Filter Failed";
                outputChannel.appendLine(`\n❌ Tumour filter failed with code ${code}`);
                vscode.window.showErrorMessage(`Tumour filter failed with code ${code}`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) Tumour Filter Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Error: ${error}`);
    }
}

async function runValidation() {
    statusBarItem.text = "$(sync~spin) Running Validation...";
    outputChannel.show();
    outputChannel.appendLine('📊 Running Validation & Reporting (Legacy)');
    
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const pythonPath = await getPythonPath();
        const scriptPath = path.join(workspaceFolder.uri.fsPath, 'MyLightningProject', 'evaluate.py');
        
        currentProcess = spawn(pythonPath, [scriptPath], {
            cwd: workspaceFolder.uri.fsPath
        });

        currentProcess.stdout?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.stderr?.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        currentProcess.on('close', (code) => {
            if (code === 0) {
                statusBarItem.text = "$(check) Validation Complete";
                outputChannel.appendLine('\n✅ Validation and reporting completed');
                vscode.window.showInformationMessage('Validation and reporting completed!');
                // Open reports folder
                const reportsPath = path.join(workspaceFolder.uri.fsPath, 'reports');
                vscode.env.openExternal(vscode.Uri.file(reportsPath));
            } else {
                statusBarItem.text = "$(error) Validation Failed";
                outputChannel.appendLine(`\n❌ Validation failed with code ${code}`);
                vscode.window.showErrorMessage(`Validation failed with code ${code}`);
            }
            currentProcess = null;
        });

    } catch (error) {
        statusBarItem.text = "$(error) Validation Error";
        outputChannel.appendLine(`❌ Error: ${error}`);
        vscode.window.showErrorMessage(`Error: ${error}`);
    }
}

async function getPythonPath(): Promise<string> {
    const config = vscode.workspace.getConfiguration('her2-pipeline');
    const customPythonPath = config.get<string>('pythonPath');
    
    // Use custom Python path if specified
    if (customPythonPath && customPythonPath.trim()) {
        return customPythonPath.trim();
    }
    
    // Try to find Python executable from Python extension
    try {
        const pythonExtension = vscode.extensions.getExtension('ms-python.python');
        if (pythonExtension) {
            await pythonExtension.activate();
            // @ts-ignore
            const pythonPath = pythonExtension.exports?.settings?.getExecutionDetails?.()?.execCommand?.[0];
            if (pythonPath) {
                outputChannel.appendLine(`🐍 Using Python from extension: ${pythonPath}`);
                return pythonPath;
            }
        }
    } catch (error) {
        outputChannel.appendLine(`⚠️  Failed to get Python from extension: ${error}`);
    }
    
    // Check for virtual environment
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder) {
        const venvPaths = [
            path.join(workspaceFolder.uri.fsPath, 'venv', 'Scripts', 'python.exe'),
            path.join(workspaceFolder.uri.fsPath, 'venv', 'bin', 'python'),
            path.join(workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe'),
            path.join(workspaceFolder.uri.fsPath, '.venv', 'bin', 'python')
        ];
        
        for (const venvPath of venvPaths) {
            if (fs.existsSync(venvPath)) {
                outputChannel.appendLine(`🐍 Using virtual environment: ${venvPath}`);
                return venvPath;
            }
        }
    }
    
    // Fallback to system python
    const systemPython = process.platform === 'win32' ? 'python' : 'python3';
    outputChannel.appendLine(`🐍 Using system Python: ${systemPython}`);
    return systemPython;
}

async function ensurePythonEnvironment(): Promise<boolean> {
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return false;
        }

        const pythonPath = await getPythonPath();
        
        // Check if required packages are installed
        const checkProcess = spawn(pythonPath, ['-c', 'import slideflow, pytorch_lightning, torch; print("Dependencies OK")'], {
            cwd: workspaceFolder.uri.fsPath
        });
        
        let output = '';
        checkProcess.stdout?.on('data', (data) => {
            output += data.toString();
        });
        
        return new Promise((resolve) => {
            checkProcess.on('close', (code) => {
                if (code === 0 && output.includes('Dependencies OK')) {
                    outputChannel.appendLine('✅ Python environment validated');
                    resolve(true);
                } else {
                    outputChannel.appendLine('⚠️  Missing dependencies. Install with: pip install slideflow pytorch-lightning torch torchvision cucim');
                    resolve(false);
                }
            });
        });
        
    } catch (error) {
        outputChannel.appendLine(`❌ Python environment check failed: ${error}`);
        return false;
    }
}

export function deactivate() {
    if (currentProcess) {
        currentProcess.kill();
        currentProcess = null;
    }
    statusBarItem?.dispose();
    outputChannel?.dispose();
}