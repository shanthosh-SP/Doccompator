import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent {

  form: FormGroup;

  constructor(
    private router: Router,
    private fb: FormBuilder,
  ){

    this.form = this.fb.group({
      username: ['', Validators.required],
      password: ['', Validators.required],
    });

  }


  onSubmit(form: FormGroup) {
    console.log(form.value);
    if(form.value.username == 'ngdemo1@tvsnext.com' && form.value.password == 'ngdemo@123'){
      this.rediectToHome();
    }

    

  }

  rediectToHome() {
    // Use the Router service to navigate to another route
    this.router.navigate(['module/home']);
  }
}
